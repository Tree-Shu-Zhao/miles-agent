#!/usr/bin/env python3
"""WebShop engine server (py3.10 tier).

Two backends, selected by --engine-workers:

- ``--engine-workers 0`` (shared): ONE ``SimServer`` behind an ``RLock``.
  Every engine op serializes on the lock — the vendored stack is thread-unsafe
  — so throughput is single-engine. Test / fallback / tiny-catalog mode.
- ``--engine-workers K`` (pool, PRODUCTION default): K worker PROCESSES, each
  its own engine, hosting many lightweight per-episode envs; K x parallel
  Lucene search. See _engine_worker.py and docs/webshop_integration.md.

Runs in the micromamba py3.10 env (JDK on PATH). The py3.13 webshop_env
adapter proxies to it; this server owns env mechanics only, reward remap lives
in recipes/webshop/reward.py (this returns the RAW score as info.task_score).

Flag defaults are for manual runs and must equal env_servers/webshop/defaults.sh
(launch_engine.sh passes everything explicitly). Only WEBSHOP_VENDOR and
WEBSHOP_DATA_DIR are read from the environment.
"""

import argparse
import logging
import os
import threading

from _webshop_engine import (
    DEFAULT_OBSERVATION_MODE,
    GOAL_SHUFFLE_SEED,
    OBSERVATION_MODES,
    EngineConfig,
    GoalTable,
    build_engine,
    drop_engine_state,
    goal_index,
    make_env,
    step_env,
)
from session_server import ResetResult, SessionBackend, StepResult, UnknownSession, base_arg_parser, serve

logger = logging.getLogger("webshop_engine")

# Layout defaults = defaults.sh with WEBSHOP_HOME unset and no MILES_AGENT_DATA:
# WEBSHOP_HOME=/data/webshop, WEBSHOP_VENDOR=$WEBSHOP_HOME/webshop (the writable
# copy setup.sh builds the catalog + index into; the committed tree next to this
# file has neither), WEBSHOP_DATA_DIR=$WEBSHOP_VENDOR/data. Pinned by
# tests/test_server.py::test_arg_parser_defaults_match_defaults_sh.
DEFAULT_HOME = "/data/webshop"
DEFAULT_VENDORED = f"{DEFAULT_HOME}/webshop"
DEFAULT_DATA_DIR = f"{DEFAULT_VENDORED}/data"


class WebshopSharedBackend(SessionBackend):
    """Single shared engine + one RLock (engine_workers=0). All engine access
    serializes on the lock (the vendored stack is single-threaded)."""

    def __init__(self, cfg: EngineConfig):
        self._lock = threading.RLock()
        self._sessions: dict[str, object] = {}
        self._cfg = cfg
        logger.info("building shared WebShop engine (%s) ...", cfg.protocol())
        self._engine, self._env_cls, self._file_paths = build_engine(cfg)
        self._goals = GoalTable(self._engine.goals)
        logger.info(
            "engine ready: %d goals, goals_digest %s (%s)", len(self._goals), self._goals.digest, cfg.protocol()
        )

    def reset(self, session_id: str, task_id, split: str, goal_id: str | None = None) -> ResetResult:
        # goal_id (content address) wins over the legacy task_id position; an
        # unknown id / bad index is a ValueError -> HTTP 400 (never retried).
        goal_idx = self._goals.resolve(goal_id) if goal_id is not None else goal_index(task_id, len(self._goals))
        with self._lock:
            old = self._sessions.pop(session_id, None)
            if old is not None:
                drop_engine_state(self._engine, getattr(old, "session_prefix", None))
            env, obs, info = make_env(
                self._engine,
                self._env_cls,
                self._file_paths,
                goal_idx,
                f"{session_id}:",
                observation_mode=self._cfg.observation_mode,
            )
            self._sessions[session_id] = env
        return ResetResult(obs=obs, info=info)

    def step(self, session_id: str, action: str) -> StepResult:
        with self._lock:
            env = self._sessions.get(session_id)
            if env is None:
                raise UnknownSession(session_id)
            obs, reward, done, info = step_env(self._engine, env, action)
            if done:
                self._sessions.pop(session_id, None)
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def close(self, session_id: str) -> None:
        with self._lock:
            env = self._sessions.pop(session_id, None)
            if env is not None:
                drop_engine_state(self._engine, getattr(env, "session_prefix", None))

    def live_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions)

    def worker_count(self) -> int:
        return 0

    def health(self) -> dict:
        return {
            "workers": 0,
            "dead_workers": 0,
            "respawns": 0,
            "goals": len(self._goals),
            "unique_goals": self._goals.unique,
            "goals_digest": self._goals.digest,
        }

    def export_goals(self) -> list[dict]:
        return self._goals.export()


def build_arg_parser() -> argparse.ArgumentParser:
    """Full server CLI: the shared flags of ``base_arg_parser`` plus the engine
    ones. The WEBSHOP_VENDOR / WEBSHOP_DATA_DIR fallbacks are read at
    parser-build time (so the parity test can pin the environment)."""
    parser = base_arg_parser("WebShop engine server (shared or engine-worker-pool)")
    parser.add_argument("--small", action="store_true", help="use the 1000-item catalog (items_*_1000.json)")
    parser.add_argument("--full", action="store_true", help="use the full catalog (overrides --small)")
    parser.add_argument("--human-goals", action="store_true", help="human-written goals (default: synthetic)")
    parser.add_argument(
        "--goal-seed",
        type=int,
        default=GOAL_SHUFFLE_SEED,
        help="seed of the whole engine build: product prices, each goal's price cap (hence its instruction_text and "
        "goal_id) and the goal list order. Default 233 = the original WebShop / TCOD constant, the value "
        "the frozen goals export (digest 455c9eaa96191605) was made under; any other value is a different goal SET and "
        "the goals_digest gate refuses it unless the goals file is re-exported (export_goals.py)",
    )
    parser.add_argument(
        "--observation-mode",
        choices=OBSERVATION_MODES,
        default=DEFAULT_OBSERVATION_MODE,
        help="vendored env page rendering: text_rich (TCOD / original baselines, default) or text (verl-agent); "
        "the adapter must be configured with the same mode",
    )
    parser.add_argument(
        "--engine-workers",
        type=int,
        default=8,
        help="engine-hosting worker processes; 0 = single shared engine + lock",
    )
    parser.add_argument(
        "--engine-call-timeout",
        type=float,
        default=120.0,
        help="per-IPC-op budget (pool mode); a worker exceeding it is killed and respawned. The adapter's "
        "engine_timeout_s (240) should exceed this plus the respawn margin (kill/join + replacement boot) so it "
        "normally sees a definitive 410/500 rather than its own ReadTimeout (same 502 outcome either way)",
    )
    parser.add_argument(
        "--vendored-path",
        default=os.environ.get("WEBSHOP_VENDOR", DEFAULT_VENDORED),
        help=f"WebShop tree with the built index, i.e. setup.sh's writable copy (default: $WEBSHOP_VENDOR or "
        f"{DEFAULT_VENDORED})",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("WEBSHOP_DATA_DIR", DEFAULT_DATA_DIR),
        help=f"dir holding items_*.json (default: $WEBSHOP_DATA_DIR or {DEFAULT_DATA_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    # Configure logging BEFORE building anything: engine build / pool boot /
    # respawn diagnostics all go through module loggers.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    small = not args.full  # --small is the default; --full flips it
    cfg = EngineConfig(
        vendored_path=args.vendored_path,
        data_dir=args.data_dir,
        small=small,
        human_goals=args.human_goals,
        seed=args.goal_seed,
        observation_mode=args.observation_mode,
    )

    if args.engine_workers > 0:
        from _engine_worker import WebshopEngineWorkerPool

        logger.info("starting %d engine workers (%s) ...", args.engine_workers, cfg.protocol())
        backend: SessionBackend = WebshopEngineWorkerPool(cfg, args.engine_workers, args.engine_call_timeout)
        logger.info("engine pool ready: %s", backend.health())
    else:
        backend = WebshopSharedBackend(cfg)

    # The goal-set digest is what the inner script gates the frozen dataset
    # against; log it once so a refused run can be diagnosed from the engine log.
    health = backend.health()
    logger.info("goal set: %d goals, goals_digest %s", health["goals"], health["goals_digest"])

    # /health carries the protocol block (seed / observation_mode / human_goals /
    # catalog) plus goals / goals_digest from the backend so the training entry
    # can refuse an engine that disagrees with the adapter config and the recipe
    # data it is about to serve.
    serve(backend, args, extra_health={"protocol": cfg.protocol()})


if __name__ == "__main__":
    main()
