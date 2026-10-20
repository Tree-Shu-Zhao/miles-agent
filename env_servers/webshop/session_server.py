"""Generic session-based env-server framework (py3.10 WebShop engine tier).

A minimal HTTP layer + backend interface for the WebShop engine: WebShop's
parallelism is engine-per-worker-many-envs (implemented in server.py /
_engine_worker.py), not one-env-per-worker. Runs OUTSIDE the nemo-gym py3.13
world, in the micromamba py3.10 env with the JDK on PATH; fastapi + uvicorn
are the only extra deps beyond the vendored WebShop stack.

Protocol consumed by the py3.13 webshop_env adapter (gym_root/resources_servers/
webshop_env/_engine_client.py):

    POST /reset  {"session_id", "goal_id" | "task_id", "split"} -> {"obs", "info"}
    POST /step   {"session_id", "action"}            -> {"obs", "reward", "done", "info"}
    POST /close  {"session_id"}                      -> {}
    GET  /health -> {"status", "sessions", "workers", "dead_workers", "respawns",
                     "goals" + "goals_digest" (engine backends), "sweeper_alive"}
    GET  /goals  -> [{"goal_id", "goal_idx", <goal content fields>}, ...] in
                    engine list order (engine backends; 501 otherwise)

/reset addresses the goal by ``goal_id`` (content id, see
agent_utils/webshop_goals.py) when present, else by the legacy ``task_id``
list position. The kwarg reaches ``backend.reset`` only when the request
carries it, so a backend with the legacy 3-argument signature keeps working
for legacy requests.

Status mapping: a ``ValueError`` from ``backend.reset`` (non-int task_id, goal
index out of range, ``UnknownGoal``: goal_id not in the engine's list) is the
client's fault -> 400 (the adapter never retries 4xx); ``UnknownSession`` from
``backend.step`` -> 410 (the episode is gone: worker respawn / TTL evict /
already terminal); anything else -> 500 with a server-side traceback (the
adapter retries /reset 5xx a bounded number of times and treats a failed
/step as a transport failure).

Robustness contract: the adapter's oversampling / weight-sync aborts abandon
episodes without closing them — sessions are TTL-evicted by the sweeper;
/reset with an existing session_id replaces it; /step on an unknown session
returns 410 (the adapter treats 4xx as a fatal drop for that episode). The TTL
table only tracks episodes the backend still holds: a terminal step (the
backend drops a done episode itself and the adapter sends no /close after it)
and a 410 both forget the id, so a "TTL-evicting session" log line always
means a genuinely abandoned episode, never a finished one.
"""

import abc
import argparse
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("webshop_engine")


@dataclass
class ResetResult:
    obs: str
    info: dict


@dataclass
class StepResult:
    obs: str
    reward: float
    done: bool
    info: dict


class UnknownSession(KeyError):
    pass


class SessionBackend(abc.ABC):
    """Synchronous backend interface; the HTTP layer handles concurrency."""

    @abc.abstractmethod
    def reset(self, session_id: str, task_id: Any, split: str, goal_id: str | None = None) -> ResetResult: ...

    @abc.abstractmethod
    def step(self, session_id: str, action: str) -> StepResult: ...

    @abc.abstractmethod
    def close(self, session_id: str) -> None: ...

    def live_sessions(self) -> list[str]:
        return []

    def worker_count(self) -> int:
        return 0

    def health(self) -> dict[str, Any]:
        """Extra /health fields. Pools report dead workers / respawns, engine
        backends add ``goals`` and ``goals_digest`` (the inner script's gate
        compares the digest with the frozen dataset's)."""
        return {"workers": self.worker_count(), "dead_workers": 0, "respawns": 0}

    def export_goals(self) -> list[dict[str, Any]]:
        """The backend's goal list as content records, for ``GET /goals``.
        Engine backends override; the default is the 501 of a backend that has
        no goal list to freeze."""
        raise NotImplementedError("this backend has no goal list")

    def shutdown(self) -> None:  # noqa: B027
        pass


def create_app(
    backend: SessionBackend,
    session_ttl: float = 1800.0,
    max_concurrency: int = 256,
    sweep_interval: float = 60.0,
    extra_health: dict[str, Any] | None = None,
):
    """FastAPI app over ``backend``. ``sweep_interval <= 0`` disables the TTL
    sweeper thread (tests drive ``app.state.sweep_once(now)`` directly).
    ``extra_health`` is merged verbatim into every /health reply (the engine
    server puts its ``protocol`` block there for the training entry's parity gate)."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class ResetRequest(BaseModel):
        session_id: str
        task_id: Any = None  # legacy: goal list position
        split: str = "train"
        goal_id: str | None = None  # content id; wins over task_id when present

    class StepRequest(BaseModel):
        session_id: str
        action: str

    class CloseRequest(BaseModel):
        session_id: str

    app = FastAPI()
    last_seen: dict[str, float] = {}
    ttl_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="webshop-http")

    def touch(session_id: str) -> None:
        with ttl_lock:
            last_seen[session_id] = time.monotonic()

    def forget(session_id: str) -> None:
        with ttl_lock:
            last_seen.pop(session_id, None)

    def sweep_once(now: float | None = None) -> list[str]:
        """Evict every session idle for more than ``session_ttl``; returns the
        evicted ids. ``now`` is injectable so tests need not sleep."""
        now = time.monotonic() if now is None else now
        with ttl_lock:
            stale = [sid for sid, t in last_seen.items() if now - t > session_ttl]
        evicted: list[str] = []
        for sid in stale:
            # Re-check under the lock: an op may have touched this session
            # between the snapshot and now. Drop the marker while holding the
            # lock so a step racing in cannot re-touch a session we then
            # close (the small residual window — step landing after this del
            # but before backend.close — self-heals: the next step 410s).
            with ttl_lock:
                t = last_seen.get(sid)
                if t is None or now - t <= session_ttl:
                    continue
                del last_seen[sid]
            logger.info("TTL-evicting session %s", sid)
            try:
                backend.close(sid)
            except Exception:
                logger.exception("TTL eviction failed for %s", sid)
            evicted.append(sid)
        return evicted

    def sweep_loop() -> None:
        while True:
            time.sleep(sweep_interval)
            try:
                sweep_once()
            except Exception:  # a failing pass must not kill the thread
                logger.exception("TTL sweep pass failed")

    sweeper: threading.Thread | None = None
    if sweep_interval > 0:
        sweeper = threading.Thread(target=sweep_loop, name="webshop-ttl-sweeper", daemon=True)
        sweeper.start()
    # Exposed for tests / operators: drive a sweep pass without waiting, and
    # inspect the TTL table.
    app.state.sweep_once = sweep_once
    app.state.last_seen = last_seen

    async def run_sync(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)

    @app.post("/reset")
    async def reset(req: ResetRequest):
        if req.goal_id is None:
            call = (backend.reset, req.session_id, req.task_id, req.split)
        else:
            # Keyword only when present: a legacy 3-argument backend never sees it.
            call = (lambda: backend.reset(req.session_id, req.task_id, req.split, goal_id=req.goal_id),)
        try:
            result = await run_sync(*call)
        except ValueError as e:
            # Bad goal index / non-int task_id / unknown goal_id: the client's
            # fault, and a retry would fail identically, so 4xx (the adapter
            # does not retry 4xx).
            logger.warning("reset rejected for %s: %s", req.session_id, e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("reset failed for %s", req.session_id)
            raise HTTPException(status_code=500, detail=repr(e)) from e
        touch(req.session_id)
        return {"obs": result.obs, "info": result.info}

    @app.post("/step")
    async def step(req: StepRequest):
        try:
            result = await run_sync(backend.step, req.session_id, req.action)
        except UnknownSession as e:
            # The backend no longer holds this episode (worker respawn / TTL /
            # already terminal): drop our marker too, or the sweeper would
            # later "evict" it and close a session that is already gone.
            forget(req.session_id)
            logger.info("step on unknown session %s -> 410", req.session_id)
            raise HTTPException(status_code=410, detail=f"unknown session {req.session_id}") from e
        except Exception as e:
            logger.exception("step failed for %s", req.session_id)
            raise HTTPException(status_code=500, detail=repr(e)) from e
        if result.done:
            # Backends drop a finished episode themselves and the adapter
            # deliberately sends no /close after a terminal step, so the id
            # must leave the TTL table now; otherwise every finished episode
            # resurfaces session_ttl later as a spurious "TTL-evicting session"
            # line (plus a backend.close on a ghost), indistinguishable from a
            # real leak.
            forget(req.session_id)
        else:
            touch(req.session_id)
        return {"obs": result.obs, "reward": result.reward, "done": result.done, "info": result.info}

    @app.post("/close")
    async def close(req: CloseRequest):
        try:
            await run_sync(backend.close, req.session_id)
        except Exception:
            logger.exception("close failed for %s", req.session_id)
        forget(req.session_id)
        return {}

    @app.get("/goals")
    async def goals():
        # The engine's goal list as content records, in list order: the source
        # env_servers/webshop/export_goals.py freezes recipes/webshop/data from.
        try:
            return await run_sync(backend.export_goals)
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            logger.exception("goal export failed")
            raise HTTPException(status_code=500, detail=repr(e)) from e

    @app.get("/health")
    async def health():
        sessions = await run_sync(backend.live_sessions)
        return {
            "status": "ok",
            "sessions": len(sessions),
            **backend.health(),
            **(extra_health or {}),
            "sweeper_alive": sweeper is not None and sweeper.is_alive(),
        }

    return app


def serve(backend: SessionBackend, args: argparse.Namespace, extra_health: dict[str, Any] | None = None) -> None:
    import uvicorn

    app = create_app(
        backend,
        session_ttl=args.session_ttl,
        max_concurrency=args.max_concurrency,
        sweep_interval=args.sweep_interval,
        extra_health=extra_health,
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        backend.shutdown()


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    """Shared flags. Defaults exist for manual runs and MUST equal
    env_servers/webshop/defaults.sh (the single source of truth for launches;
    launch_engine.sh passes every value explicitly)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="127.0.0.1", help="loopback only: the adapter runs on the same node")
    parser.add_argument("--port", type=int, default=6200, help="below a 9000+ ephemeral port range")
    parser.add_argument(
        "--session-ttl",
        type=float,
        default=2400.0,
        help="idle seconds before a session is evicted; must exceed the adapter's session_ttl_s (1800) "
        "so the engine never evicts a session the adapter still considers live",
    )
    parser.add_argument("--max-concurrency", type=int, default=256, help="HTTP worker threads for backend calls")
    parser.add_argument(
        "--sweep-interval", type=float, default=60.0, help="seconds between TTL sweeps; <= 0 disables the sweeper"
    )
    return parser
