"""Process slots: N spawn'd worker processes, one Pipe + one single-thread
executor each, shared by the env tiers that host state in child processes.

Two facades sit on top of this module and stay thin:

- ``gym_root/resources_servers/alfworld_env/_worker.py`` (py3.13, ASYNC facade,
  one TextWorld env per remote handle, lazily resolved factory), and
- ``env_servers/webshop/_engine_worker.py`` (py3.10, SYNC facade, one engine
  per child hosting many episodes, engine built eagerly before ready).

Because the webshop engine tier runs in a py3.10 micromamba env, this module is
stdlib-only and sticks to py3.10 syntax (``X | None`` annotations are fine; no
``match``, no PEP 695 generics, no ``type`` statements).

Child side: ``serve_loop`` — a single-threaded request/reply loop over one Pipe.
Single-threaded means strictly sequential ops per child, which is exactly the
property both hosted stacks need (TextWorld's thread-unsafe translate, WebShop's
module-global Flask/random state) without any locking.

Parent side: ``ProcSlots``. Every op on slot i executes on slot i's dedicated
single-thread executor, which is THE serializer for that slot's pipe (no
per-slot lock): callers queue FIFO behind each other, a wedged slot delays only
callers pinned to it, and async callers hold no thread while waiting.

Failure model: an op exceeding ``call_timeout_s``, or a dead pipe, SIGKILLs and
respawns the child (it may be wedged in C code). Everything resident in that
child is gone; the per-slot generation counter makes stale handles raise
``SlotDied`` instead of touching a stranger's state after the respawn. A
stale generation is rejected O(1) at submit time; ops already queued on that
slot's executor fail the generation check as soon as they reach it. Ops that
fail while the child stays alive raise ``SlotOpError`` and cost nothing else.

Boot handshake: each child sends one ready message (optionally carrying a
payload, e.g. the goal count) before serving; ``ProcSlots.__init__`` starts all
children first, then awaits every handshake, so a deploy/venv/data breakage
fails construction loudly with the real cause instead of degrading into a
silent per-op respawn storm.

Dead-slot parking + revival: a slot is parked (``dead_slots``, count
``DEAD_SLOT``) for the whole duration of a respawn and stays parked when the
replacement fails to boot (then also in ``failed_slots``), so the least-loaded
picker stops routing new work to it and ops pinned to it fail fast — otherwise
every op would re-enter respawn, a ``spawn_ready_timeout_s``-per-op storm. A
daemon thread retries FAILED slots every ``revive_interval_s``; each retry runs
on the slot's own executor and re-checks ``failed_slots`` there, so it never
races an in-flight call and never re-kills a replacement that a normal respawn
(or an earlier revive) already booted. A retry that boots unparks the slot.

Known accepted gap (documented once, here): ``call_timeout_s`` bounds
time-to-first-byte (``conn.poll``); the ``conn.recv`` that follows has no
timeout, so a child frozen-alive (SIGSTOP, cgroup freezer) in the microseconds
between reply header and body would hang its slot thread past the budget.
Externally-frozen-mid-write is the only trigger; a dead child raises EOFError
and respawns normally.
"""

import asyncio
import logging
import multiprocessing as mp
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

# Wire markers. READY is the first message of every child; UNKNOWN is the
# (False, payload) a handler produces by returning ``serve_loop.UNKNOWN`` —
# "the target of this op no longer exists here" (webshop: episode closed/
# evicted between the parent pinning it and the op arriving -> HTTP 410).
READY = "__ready__"
UNKNOWN = "__unknown__"
# Count sentinel parking a slot whose respawn failed to boot; ``dead_slots`` is
# the authoritative membership test, the huge count just keeps ``min`` away.
DEAD_SLOT = 10**9

_UNKNOWN_SENTINEL = object()


class SlotOpError(RuntimeError):
    """The op failed inside the child; the child itself is still healthy."""


class SlotDied(RuntimeError):
    """The child was killed (op timeout / crash / respawn); its resident state is gone."""


def _send(conn, payload) -> bool:
    try:
        conn.send(payload)
    except (BrokenPipeError, OSError):
        return False
    return True


def serve_loop(
    conn,
    handlers: dict[str, Callable[..., Any]],
    ready_payload: Any = None,
    *,
    build: Callable[[], Any] | None = None,
) -> None:
    """Child side: serve ``(op, *args)`` messages until EOF or ``("shutdown",)``.

    ``build`` (optional) runs BEFORE the ready handshake and its return value
    becomes the ready payload; if it raises, the child reports
    ``(False, "ExcType: msg\n<traceback>")`` on the pipe and exits (the parent
    turns that into a loud boot failure carrying the child-side traceback).
    Without ``build`` the child is ready at once and ``ready_payload`` is sent
    as-is (alfworld: lazy factory, payload None).

    Per message: ``handlers[op](*args)`` -> ``(True, result)``; an exception ->
    ``(False, "ExcType: msg")`` with the traceback printed to the child's stderr
    (never die silently mid-protocol); a handler
    returning ``serve_loop.UNKNOWN`` -> ``(False, UNKNOWN)``. ``shutdown``
    replies ``(True, None)`` and exits.
    """
    if build is not None:
        try:
            ready_payload = build()
        except BaseException as e:  # data/JVM/venv breakage: report, then die
            _send(conn, (False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
            return
    if not _send(conn, (True, (READY, ready_payload))):
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):  # parent died or respawned us
            return
        op, args = msg[0], msg[1:]
        if op == "shutdown":
            _send(conn, (True, None))
            return
        try:
            handler = handlers.get(op)
            if handler is None:
                raise ValueError(f"unknown op {op!r}")
            result = handler(*args)
        except BaseException as e:
            traceback.print_exc()  # child stderr = the worker/engine log
            reply: tuple = (False, f"{type(e).__name__}: {e}")
        else:
            reply = (False, UNKNOWN) if result is _UNKNOWN_SENTINEL else (True, result)
        if not _send(conn, reply):
            return


serve_loop.UNKNOWN = _UNKNOWN_SENTINEL  # type: ignore[attr-defined]


class ProcSlots:
    """N spawn'd children, one Pipe + one dedicated single-thread executor each.

    Thread-safety invariants:
    - ``_gens[i]``, ``_procs[i]``, ``_conns[i]``, ``ready_payloads[i]`` are
      mutated only on slot i's own executor thread after ``__init__`` (single
      writer), by ``_call_sync`` -> ``_respawn`` or by ``revive_dead_slots``.
    - ``_counts``, ``dead_slots``, ``respawns`` and the gen bump in ``_respawn``
      happen under ``lock`` (an RLock, public so facades can make compound
      decisions atomic, e.g. "gen still current -> register the session").
    - ``on_respawn(idx, new_gen)`` (optional) is invoked under ``lock`` right
      after the gen bump, so a facade can drop everything it had pinned to
      that slot in the same critical section.
    """

    def __init__(
        self,
        n: int,
        call_timeout_s: float,
        target: Callable[..., None],
        args_for_slot: Callable[[int], tuple],
        *,
        name: str = "slot",
        spawn_ready_timeout_s: float = 120.0,
        revive_interval_s: float = 60.0,
        op_error_cls: type[SlotOpError] = SlotOpError,
        died_cls: type[SlotDied] = SlotDied,
        on_respawn: Callable[[int, int], None] | None = None,
        validate_ready: Callable[[int, Any], None] | None = None,
    ):
        assert n > 0
        self._ctx = mp.get_context("spawn")
        self._n = n
        self._timeout = call_timeout_s
        self._target = target
        self._args_for_slot = args_for_slot
        self._name = name
        self._spawn_ready_timeout_s = spawn_ready_timeout_s
        self._op_error_cls = op_error_cls
        self._died_cls = died_cls
        self._on_respawn = on_respawn
        # Optional (slot idx, ready payload) check run inside the boot
        # handshake, at first boot AND after every respawn; a RuntimeError it
        # raises fails that boot like a child build error (the slot is parked
        # in failed_slots and retried), so a replacement child that came up
        # with different data never serves.
        self._validate_ready = validate_ready
        self._procs: list[Any] = [None] * n
        self._conns: list[Any] = [None] * n
        self.ready_payloads: list[Any] = [None] * n
        self._executors = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{name}-ipc-{i}") for i in range(n)]
        self._gens = [0] * n
        self._counts = [0] * n
        self.lock = threading.RLock()
        self.respawns = 0
        self.dead_slots: set[int] = set()  # parked: respawn in flight OR failed
        self.failed_slots: set[int] = set()  # subset of dead_slots: last boot failed, awaiting revival
        self.closing = False
        self._revive_stop = threading.Event()
        self._revive_thread: threading.Thread | None = None
        # Two-phase boot: start every child first (imports/builds run in
        # parallel), then wait for all ready handshakes. Any failure kills every
        # child we started (they are daemons, but the parent may live on, e.g.
        # a test process) and propagates the real cause.
        for idx in range(n):
            self._start_proc(idx)
        try:
            for idx in range(n):
                self._await_ready(idx)
        except BaseException:
            for proc in self._procs:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise
        logger.info("%s pool ready: %d slots, call timeout %.0fs", name, n, call_timeout_s)
        if revive_interval_s > 0:
            self._revive_thread = threading.Thread(
                target=self._revive_loop, args=(revive_interval_s,), name=f"{name}-revive", daemon=True
            )
            self._revive_thread.start()

    # ── process lifecycle (each idx mutated only from its own executor thread after __init__) ──

    def _start_proc(self, idx: int) -> None:
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=self._target,
            args=(child_conn, *self._args_for_slot(idx)),
            name=f"{self._name}-{idx}",
            daemon=True,  # die with the server; no graceful drain needed
        )
        proc.start()
        child_conn.close()  # so a dead child is detectable via EOF
        self._procs[idx] = proc
        self._conns[idx] = parent_conn

    def _await_ready(self, idx: int) -> None:
        conn = self._conns[idx]
        try:
            if not conn.poll(self._spawn_ready_timeout_s):
                raise TimeoutError(f"no ready handshake within {self._spawn_ready_timeout_s}s")
            ok, payload = conn.recv()
            if not ok:
                raise RuntimeError(f"{self._name} {idx} failed to build: {payload!r}")
            if not (isinstance(payload, tuple) and len(payload) == 2 and payload[0] == READY):
                raise RuntimeError(f"unexpected handshake reply: {(ok, payload)!r}")
            if self._validate_ready is not None:
                self._validate_ready(idx, payload[1])
            self.ready_payloads[idx] = payload[1]
        except (EOFError, OSError, TimeoutError, RuntimeError) as e:
            # The child died during bootstrap (venv/deploy/data breakage): fail
            # loudly with the cause instead of a silent per-op respawn storm.
            try:
                self._procs[idx].kill()
            except Exception:
                pass
            # A build failure embeds the child-side traceback in ``e``; a child
            # that died before the handshake left its traceback in its stderr.
            hint = "" if isinstance(e, RuntimeError) else "; the child died before the handshake, see its stderr"
            raise RuntimeError(f"{self._name} {idx} failed to boot ({e}){hint}") from e
        logger.info("%s %d ready (pid %s, gen %d)", self._name, idx, self._procs[idx].pid, self._gens[idx])

    def _respawn(self, idx: int, reason: str = "") -> None:
        """Kill + reboot slot idx. Runs on slot idx's own executor thread.

        Parks the slot BEFORE the reboot so the picker skips it while the
        replacement boots; reservations made on the NEW gen meanwhile queue on
        the executor and are kept when the slot unparks. Raises ``died_cls``
        (leaving the slot parked and marked failed) when the replacement fails
        to start or boot.
        """
        if self.closing:
            raise self._died_cls(f"{self._name} {idx}: pool is shutting down")
        with self.lock:
            self.respawns += 1
            self._gens[idx] += 1
            gen = self._gens[idx]
            self._counts[idx] = DEAD_SLOT
            self.dead_slots.add(idx)
            if self._on_respawn is not None:
                self._on_respawn(idx, gen)
        logger.warning("%s %d respawning -> gen %d (respawns=%d): %s", self._name, idx, gen, self.respawns, reason)
        try:
            self._procs[idx].kill()
            self._procs[idx].join(5)
        except Exception:
            pass
        try:
            self._conns[idx].close()
        except Exception:
            pass
        try:
            # Process.start() can itself fail (EMFILE, fork/exec failure): same
            # parked-until-revival path as a replacement that never handshakes.
            self._start_proc(idx)
            self._await_ready(idx)
        except Exception as e:
            # Replacement could not boot (transient OOM/disk, or a now-
            # persistent breakage). Leave the slot parked: the balancer stops
            # routing to it and ops pinned to it fail fast; the revive thread
            # retries periodically. Surface as died_cls so the caller drops the
            # episode (webshop: 410) rather than 500.
            with self.lock:
                self.failed_slots.add(idx)
            logger.warning("%s %d failed to respawn; parked until revival: %s", self._name, idx, e)
            raise self._died_cls(f"{self._name} {idx} failed to respawn: {e}") from e
        with self.lock:
            self._counts[idx] = max(0, self._counts[idx] - DEAD_SLOT)
            self.dead_slots.discard(idx)
            self.failed_slots.discard(idx)
        logger.info("%s %d respawned (gen %d)", self._name, idx, gen)

    # ── placement bookkeeping (all under ``lock``) ─────────────────────

    def gen(self, idx: int) -> int:
        with self.lock:
            return self._gens[idx]

    def reserve(self) -> tuple[int, int]:
        """Pick the least-loaded live slot (parked slots only when every slot is
        parked) and count the reservation. Returns ``(idx, gen)``; pair with
        ``rollback`` on failure or ``release`` when the resident state is gone."""
        with self.lock:
            live = [i for i in range(self._n) if i not in self.dead_slots]
            idx = min(live or range(self._n), key=lambda i: self._counts[i])
            self._counts[idx] += 1
            return idx, self._gens[idx]

    def release(self, idx: int, gen: int) -> bool:
        """Drop one reservation IF ``gen`` is still current (a respawn in between
        already discarded the whole old-gen count). Returns whether it was."""
        with self.lock:
            if gen != self._gens[idx]:
                return False
            self._counts[idx] = max(0, self._counts[idx] - 1)
            return True

    def rollback(self, idx: int, gen: int) -> bool:
        """Undo a ``reserve`` whose first op failed (same bookkeeping as ``release``)."""
        return self.release(idx, gen)

    # ── calls ─────────────────────────────────────────────────────────

    def _precheck(self, idx: int, gen: int) -> None:
        """O(1) fail-fast shared by submit time and execution time (the latter
        is authoritative: a respawn may complete while an op waits its turn)."""
        with self.lock:
            if self.closing:
                raise self._died_cls(f"{self._name} {idx}: pool is shutting down")
            if gen != self._gens[idx]:
                raise self._died_cls(f"{self._name} {idx} was respawned; gen {gen} is gone")
            if idx in self.failed_slots:
                raise self._died_cls(f"{self._name} {idx} is parked after a failed respawn (awaiting revival)")

    def call(self, idx: int, gen: int, msg: tuple) -> Any:
        """Sync: run the op on slot idx's executor and wait for it. A stale
        gen / failed slot / closing pool fails here without queueing behind an
        in-flight respawn."""
        self._precheck(idx, gen)
        return self._executors[idx].submit(self._call_sync, idx, gen, msg).result()

    async def acall(self, idx: int, gen: int, msg: tuple) -> Any:
        """Async: same, awaiting the slot executor without holding any thread."""
        self._precheck(idx, gen)
        return await asyncio.get_running_loop().run_in_executor(self._executors[idx], self._call_sync, idx, gen, msg)

    def _call_sync(self, idx: int, gen: int, msg: tuple) -> Any:
        # Single-threaded per slot (its executor), so pipe access and
        # _gens/_procs/_conns mutation need no locking here.
        op = msg[0]
        # Re-checked here because the op may have waited behind a respawn (or
        # shutdown began meanwhile: never read a child's shutdown reply as an
        # op result).
        self._precheck(idx, gen)
        with self.lock:
            if idx in self.dead_slots:
                raise self._died_cls(f"{self._name} {idx} is parked after a failed respawn (awaiting revival)")
        conn = self._conns[idx]
        t0 = time.monotonic()
        try:
            conn.send(msg)
            if not conn.poll(self._timeout):
                raise TimeoutError(f"op {op!r} exceeded {self._timeout}s")
            ok, payload = conn.recv()
        except (BrokenPipeError, EOFError, OSError, TimeoutError) as e:
            elapsed = time.monotonic() - t0
            if self.closing:  # shutdown killed the child under us: no respawn
                raise self._died_cls(f"{self._name} {idx} closing during {op!r}: {e}") from e
            logger.warning(
                "%s %d gen %d: op %r failed after %.1fs (%s: %s); respawning",
                self._name,
                idx,
                gen,
                op,
                elapsed,
                type(e).__name__,
                e,
            )
            self._respawn(idx, reason=f"{op!r}: {type(e).__name__}: {e}")
            if isinstance(e, TimeoutError):
                raise self._died_cls(f"op {op!r} exceeded {self._timeout}s on {self._name} {idx}") from e
            raise self._died_cls(f"{self._name} {idx} died on {op!r}: {e!r}") from e
        if not ok:
            raise self._op_error_cls(payload)
        return payload

    # ── revival / health / shutdown ───────────────────────────────────

    def _revive_one(self, idx: int) -> bool:
        """Runs on slot idx's executor. Re-checks ``failed_slots`` THERE: by the
        time a queued revive runs, a normal respawn or an earlier revive may
        already have booted a healthy replacement, which must not be killed."""
        with self.lock:
            if self.closing or idx not in self.failed_slots:
                return False
        self._respawn(idx, "revive")
        return True

    def revive_dead_slots(self) -> int:
        """Retry every FAILED slot once, on that slot's own executor (so it never
        races an in-flight call). Idempotent: slots merely mid-respawn, or
        already revived by the time the retry runs, are left alone. Returns how
        many came back."""
        with self.lock:
            failed = sorted(self.failed_slots)
        revived = 0
        for idx in failed:
            if self.closing:
                break
            try:
                if self._executors[idx].submit(self._revive_one, idx).result():
                    revived += 1
            except Exception as e:  # _respawn already logged the boot failure
                logger.warning("%s %d revival failed: %s", self._name, idx, e)
        if revived:
            logger.info("%s: revived %d/%d failed slots", self._name, revived, len(failed))
        return revived

    def _revive_loop(self, interval_s: float) -> None:
        while not self._revive_stop.wait(interval_s):
            if self.closing:
                return
            if self.failed_slots:
                try:
                    self.revive_dead_slots()
                except Exception:
                    logger.exception("%s: revive pass failed", self._name)

    def health(self) -> dict[str, int]:
        with self.lock:
            return {"workers": self._n, "dead_workers": len(self.dead_slots), "respawns": self.respawns}

    def shutdown(self, grace_s: float = 5.0) -> None:
        """Stop everything: ask each child to exit, SIGKILL whatever is still
        alive after ``grace_s`` (shared deadline, so a wedged child costs at most
        grace_s total), release the executors and stop the revive thread.

        The shutdown request is written straight to the pipes rather than
        queued on the slot executors: a slot wedged in a hung op would otherwise
        delay its own shutdown by a full call_timeout_s. Its executor thread
        sees EOF once the child is killed and, with ``closing`` set, raises
        died_cls without respawning.
        """
        self.closing = True
        self._revive_stop.set()
        for conn in self._conns:
            if conn is not None:
                _send(conn, ("shutdown",))
        deadline = time.monotonic() + grace_s
        for proc in self._procs:
            if proc is None:
                continue
            proc.join(max(0.0, deadline - time.monotonic()))
            if proc.is_alive():
                proc.kill()
                proc.join(1.0)
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
        for executor in self._executors:
            executor.shutdown(wait=False, cancel_futures=True)
        if self._revive_thread is not None:
            self._revive_thread.join(1.0)
        logger.info("%s pool shut down", self._name)
