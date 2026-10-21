"""OpenSandbox (https://github.com/opensandbox-group/OpenSandbox) as the trials' sandbox backend, for `harbor run`
and the trial server alike: Harbor's own OpenSandboxEnvironment (harbor[opensandbox]) with what a shared, remote
OpenSandbox server needs.

    HARBOR_ENVIRONMENT_IMPORT_PATH=miles_harbor.opensandbox_env:OpenSandboxEnvironment

- the server and its key come from the environment, never from the kwargs (scripts/harbor_run.sh and
  harbor_trial_server.sh print the kwargs with the job's config): OPENSANDBOX_ENDPOINT (an https URL; Harbor's
  OPENSANDBOX_DOMAIN too) and OPENSANDBOX_KEY (Harbor's OPENSANDBOX_API_KEY too); an ``api_key`` kwarg is refused.
  The endpoint's scheme is every connection's (``protocol`` when it names none: https), the key's included;
- a server whose certificate no public CA signed: OPENSANDBOX_CA_FILE (PEM; the ``ca_file`` kwarg) is trusted
  besides the usual CAs (SSL_CERT_FILE, else certifi's), on this backend's connections alone; OPENSANDBOX_PROXY
  (``proxy``): the HTTP proxy to reach the server through (the SDK's own transport ignores HTTPS_PROXY);
- the server proxy is on (``use_server_proxy``): a sandbox's own address is the server's cluster's, which a client
  outside it does not reach (the SDK's health check then hangs until ``ready_timeout_sec``);
- a command's output as it printed it: execd streams one message a line, its terminator stripped (an empty line as
  "\\n"), and Harbor joins the messages with nothing; this backend puts the newlines back (joined_lines). Whether
  the output's last line had one is not known: it always gets one;
- a command that outlives its ``timeout_sec`` raises ``RuntimeError("Command timed out after <n> seconds")``, as
  Harbor's docker environment does (vanillux tells the model so, as TMax did): execd kills it, or the stream it
  answers on stays silent STREAM_GRACE_S past it;
- each sandbox carries ``miles-harbor-run=<RUN_TAG>`` (run_label) in its metadata and lives SANDBOX_TIMEOUT_S at
  most (the server's TTL; ``sandbox_timeout_sec``): a run that died leaves its sandboxes for that long, and
  ``sweep`` kills them at once;
- httpx's INFO line per request is dropped for the server's requests (QuietServerRequests): a trial makes one per
  command and each names the server; their warnings and errors stay;
- the server rate-limits creates: a create it defers answers 429 with a Retry-After, and nothing was created. Every
  connection retries a POST answered 429 after that wait (retry_policy; the SDK retries only idempotent methods by
  itself), on top of Harbor's own three create attempts. Each retry of the SDK is counted by cause, and a line
  ``OPENSANDBOX_RETRY cause=<cause> n=<since its last line> total=<n> backoff_s=<s> <method> <path>`` logs a cause
  at its first retry and then at most every RETRY_LOG_S seconds (RetryLog). Creates in flight in a process are
  bounded, each through its readiness wait (OPENSANDBOX_MAX_CREATES, default MAX_CREATES; 0: no bound).

The trial server's hooks (trial_server.Backend): ``unrunnable_tasks`` stops the server on a task OpenSandbox cannot
run (one image, prebuilt: no Dockerfile build, no compose; a network it keeps from create to delete: public or none,
the agent's and the verifier's phases the environment's) or on a missing server; ``refuse_unreachable_agent_network``
warns that ``agent_network=endpoint`` is not enforced (the agent keeps the task's network: a sandbox's egress is set
as it is created, and no runtime rule opens it again); a harness inside the sandbox reaches its endpoint only if it is
public.

    python -m miles_harbor.opensandbox_env check [--image ubuntu:24.04]   # the server, the key, a command's round trip
    python -m miles_harbor.opensandbox_env sweep --run <RUN_TAG> [--dry-run]   # kill a run's sandboxes

Neither command prints the server's address: ``<endpoint>`` stands for it in their errors and log lines.
"""

import argparse
import asyncio
import dataclasses
import logging
import os
import re
import ssl
import sys
import time
import weakref
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from harbor.environments.base import ExecResult
from harbor.environments.opensandbox import OpenSandboxEnvironment as HarborOpenSandboxEnvironment
from harbor.models.task.config import NetworkMode, TaskConfig

logger = logging.getLogger(__name__)

ENDPOINT_VARS = ("OPENSANDBOX_ENDPOINT", "OPENSANDBOX_DOMAIN")
KEY_VARS = ("OPENSANDBOX_KEY", "OPENSANDBOX_API_KEY")
CA_FILE_VAR = "OPENSANDBOX_CA_FILE"
PROXY_VAR = "OPENSANDBOX_PROXY"
MAX_CREATES_VAR = "OPENSANDBOX_MAX_CREATES"
# creates in flight at once in a process (the server asks its clients to keep them bounded and back off on 429)
MAX_CREATES = 32
# a retry cause's OPENSANDBOX_RETRY line at most this often: a burst of 429s is a line, not one per request
RETRY_LOG_S = 30
# the server's TTL for a sandbox: past every trial's timeouts (TMax: agent 1 h, verifier 10 min; trial_timeout_s 2 h)
SANDBOX_TIMEOUT_S = 6 * 3600
# how long past timeout_sec a command's stream may stay silent before exec gives up on it
STREAM_GRACE_S = 60
RUN_LABEL = "miles-harbor-run"
# kills in flight at once in a sweep: a run's sandboxes go within a step's kill grace (Slurm's KillWait)
SWEEP_CONCURRENCY = 16
COMPOSE_FILES = ("docker-compose.yaml", "docker-compose.yml")
CHECK_OUTPUT = "a\n\nb\n"  # printf's: two lines around an empty one


def first_set(names: Iterable[str]) -> str | None:
    """The value of the first of ``names`` set (non-empty) in the environment."""
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def endpoint_scheme(domain: str | None) -> str | None:
    """The scheme an endpoint URL names ("http" / "https"), or None. The SDK takes the lifecycle API's from the URL
    but the execd plane's (where the key travels too, with the server proxy) from ``protocol`` alone, http by
    default: the two must agree."""
    match = re.match(r"(https?)://", domain or "", re.I)
    return match.group(1).lower() if match else None


def endpoint_host(domain: str) -> str:
    """An endpoint URL's host[:port] (the URL as it is without a scheme)."""
    return re.sub(r"^[a-z]+://", "", domain, flags=re.I).split("/", 1)[0]


class QuietServerRequests(logging.Filter):
    """httpx's logger: its INFO lines (one per request) about ``host`` dropped, every other record kept."""

    def __init__(self, host: str) -> None:
        super().__init__()
        self.host = host

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno > logging.INFO or f"://{self.host}/" not in record.getMessage()


def quiet_server_requests(domain: str) -> None:
    """QuietServerRequests for the server at ``domain`` on httpx's logger, once per host in a process."""
    host = endpoint_host(domain)
    httpx_logger = logging.getLogger("httpx")
    if not any(isinstance(f, QuietServerRequests) and f.host == host for f in httpx_logger.filters):
        httpx_logger.addFilter(QuietServerRequests(host))


def run_label(tag: str) -> str:
    """RUN_TAG as a metadata label value (a Kubernetes one: at most 63 of [A-Za-z0-9._-], alphanumeric at both ends;
    any other character a "-"); "" for a tag with nothing left."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", tag)[:63].strip("._-")


def joined_lines(messages: Iterable[Any]) -> str:
    """A stream of execd's messages (one a line, its terminator stripped; an empty line "\\n") as the text printed."""
    return "".join(m.text if m.text.endswith("\n") else m.text + "\n" for m in messages)


class RetryLog:
    """The SDK's retries in this process by cause (its RetryCause: ``status_429``, ``status_503``...): ``total``
    counts them, and a cause's OPENSANDBOX_RETRY line is logged at its first retry, then at most every RETRY_LOG_S
    seconds with the count since its last line. The request's path is logged, never its host."""

    def __init__(self) -> None:
        self.total: Counter[str] = Counter()
        self._since: Counter[str] = Counter()
        self._logged: dict[str, float] = {}

    def note(self, event: Any) -> None:
        cause = str(getattr(event.cause, "value", event.cause))
        self.total[cause] += 1
        self._since[cause] += 1
        now = time.monotonic()
        last = self._logged.get(cause)
        if last is not None and now - last < RETRY_LOG_S:
            return
        logger.warning(
            "OPENSANDBOX_RETRY cause=%s n=%d total=%d backoff_s=%.1f %s %s",
            cause,
            self._since[cause],
            self.total[cause],
            event.backoff.total_seconds(),
            event.method,
            urlsplit(event.url).path,
        )
        self._since[cause] = 0
        self._logged[cause] = now


RETRIES = RetryLog()


def retry_policy(base: Any) -> Any:
    """``base`` (the SDK's RetryPolicy) retrying a POST or PATCH answered 429 too, after the server's Retry-After:
    the server defers a create it rate-limits that way, and nothing was done. Every retry goes to RETRIES (after
    ``base``'s own hook, if any)."""
    from http import HTTPStatus

    hook = base.on_retry

    def on_retry(event: Any) -> None:
        if hook is not None:
            hook(event)
        RETRIES.note(event)

    return dataclasses.replace(
        base,
        retryable_status_codes_non_idempotent=frozenset(base.retryable_status_codes_non_idempotent)
        | {HTTPStatus.TOO_MANY_REQUESTS},
        on_retry=on_retry,
    )


_create_slots: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def create_slots() -> asyncio.Semaphore | None:
    """The running event loop's bound on creates in flight (OPENSANDBOX_MAX_CREATES, else MAX_CREATES), or None for
    0 (no bound). The first call in a loop sets its size."""
    size = int(os.environ.get(MAX_CREATES_VAR) or MAX_CREATES)
    if size <= 0:
        return None
    loop = asyncio.get_running_loop()
    slots = _create_slots.get(loop)
    if slots is None:
        slots = _create_slots[loop] = asyncio.Semaphore(size)
    return slots


def pinned_transport(retry_policy: Any, ca_file: str | None, proxy: str | None) -> Any:
    """The SDK's default transport stack (an httpx.AsyncHTTPTransport, retry-wrapped as its policy wants, its
    limits) trusting ``ca_file`` too and going through ``proxy``. Its owner closes it: the SDK leaves a transport it
    was given alone."""
    import certifi
    import httpx
    from opensandbox.transport import RetryAsyncTransport

    context = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or certifi.where())
    if ca_file:
        if not Path(ca_file).is_file():
            raise ValueError(f"OpenSandboxEnvironment: no CA file {ca_file} ({CA_FILE_VAR} or the ca_file kwarg)")
        context.load_verify_locations(cafile=ca_file)
    inner = httpx.AsyncHTTPTransport(
        verify=context,
        proxy=proxy,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0),
    )
    if retry_policy.wraps_transport():
        return RetryAsyncTransport(inner, retry_policy, owns_inner=True)
    return inner


def connection_config(
    *, ca_file: str | None = None, proxy: str | None = None, request_timeout_sec: float | None = None
) -> tuple[Any, Any]:
    """(ConnectionConfig, its transport to close or None) for the server the environment names, as
    OpenSandboxEnvironment connects (the CLI's)."""
    from opensandbox.config import ConnectionConfig

    domain = first_set(ENDPOINT_VARS)
    fields: dict[str, Any] = {"domain": domain, "api_key": first_set(KEY_VARS), "use_server_proxy": True}
    fields["protocol"] = endpoint_scheme(domain) or "https"
    if request_timeout_sec is not None:
        fields["request_timeout"] = timedelta(seconds=request_timeout_sec)
    fields["retry_policy"] = retry_policy(ConnectionConfig(**fields).retry_policy)
    config = ConnectionConfig(**fields)
    ca_file = ca_file or os.environ.get(CA_FILE_VAR) or None
    proxy = proxy or os.environ.get(PROXY_VAR) or None
    if not (ca_file or proxy):
        return config, None
    transport = pinned_transport(config.retry_policy, ca_file, proxy)
    return ConnectionConfig(**fields, transport=transport, disable_metrics=True), transport


class OpenSandboxEnvironment(HarborOpenSandboxEnvironment):
    """Harbor's OpenSandboxEnvironment, its server from the environment (the module's docstring says what else)."""

    def __init__(
        self,
        *args: Any,
        domain: str | None = None,
        api_key: str | None = None,
        protocol: str = "https",
        use_server_proxy: bool = True,
        sandbox_timeout_sec: int = SANDBOX_TIMEOUT_S,
        metadata: dict[str, str] | None = None,
        ca_file: str | None = None,
        proxy: str | None = None,
        **kwargs: Any,
    ) -> None:
        if api_key:
            raise ValueError(f"OpenSandboxEnvironment: no api_key kwarg (the config is printed): set {KEY_VARS[0]}")
        domain = domain or first_set(ENDPOINT_VARS)
        if not domain:
            raise ValueError(f"OpenSandboxEnvironment: no server: set {' or '.join(ENDPOINT_VARS)}")
        quiet_server_requests(domain)
        self._ca_file = ca_file or os.environ.get(CA_FILE_VAR) or None
        self._proxy = proxy or os.environ.get(PROXY_VAR) or None
        self._transport: Any = None
        label = run_label(os.environ.get("RUN_TAG", ""))
        super().__init__(
            *args,
            domain=domain,
            api_key=first_set(KEY_VARS),
            protocol=endpoint_scheme(domain) or protocol,
            use_server_proxy=use_server_proxy,
            sandbox_timeout_sec=sandbox_timeout_sec,
            metadata={**({RUN_LABEL: label} if label else {}), **(metadata or {})},
            **kwargs,
        )

    @classmethod
    def preflight(cls) -> None:
        if not first_set(ENDPOINT_VARS):
            logger.warning("OpenSandbox: %s is not set: every trial will fail to start", " or ".join(ENDPOINT_VARS))

    def _build_connection_config(self, sdk: dict[str, Any]) -> Any:
        config = super()._build_connection_config(sdk)
        fields = {name: getattr(config, name) for name in config.model_fields_set}
        fields["retry_policy"] = retry_policy(config.retry_policy)
        if not (self._ca_file or self._proxy):
            return type(config)(**fields)
        # one transport for every connection of this environment (each create attempt, the sandbox's adapters),
        # closed by stop(); the config is built anew so that the SDK knows the transport is not its own to close.
        # The SDK's create-latency report has a client of its own, which could not verify this server: off.
        if self._transport is None:
            self._transport = pinned_transport(fields["retry_policy"], self._ca_file, self._proxy)
        return type(config)(**fields, transport=self._transport, disable_metrics=True)

    async def _create_sandbox(self, sdk: dict[str, Any]) -> Any:
        # Harbor's create (its three attempts and the readiness wait) holds one of the process's create slots
        slots = create_slots()
        if slots is None:
            return await super()._create_sandbox(sdk)
        async with slots:
            return await super()._create_sandbox(sdk)

    async def stop(self, delete: bool) -> None:
        try:
            await super().stop(delete)
        finally:
            transport, self._transport = self._transport, None
            if transport is not None:
                await transport.aclose()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        # Harbor's exec, its output's newlines and its timeout as Harbor's docker environment has them
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        sdk = self._load_opensandbox()
        opts = self._build_run_command_opts(sdk, cwd, self._merge_env(env), timeout_sec, user)
        timed_out = f"Command timed out after {timeout_sec} seconds"
        started = time.monotonic()
        run = self._sandbox.commands.run(command, opts=opts)
        try:
            execution = await (run if timeout_sec is None else asyncio.wait_for(run, timeout_sec + STREAM_GRACE_S))
        except TimeoutError:
            raise RuntimeError(timed_out) from None
        return_code = execution.exit_code
        if return_code is None:
            return_code = 1 if execution.error is not None else 0
        # execd's kill at the timeout: exit code -1 ("signal: killed"), told apart from another kill by the clock
        if timeout_sec is not None and return_code == -1 and time.monotonic() - started >= timeout_sec:
            raise RuntimeError(timed_out)
        return ExecResult(
            stdout=joined_lines(execution.logs.stdout),
            stderr=joined_lines(execution.logs.stderr),
            return_code=return_code,
        )


def why_unrunnable(task_dir: Path) -> str | None:
    """Why OpenSandbox cannot run the task at ``task_dir`` (what Harbor would refuse as its trial starts), or None."""
    environment = task_dir / "environment"
    if any((environment / name).exists() for name in COMPOSE_FILES):
        return "a docker-compose environment (OpenSandbox runs one prebuilt image)"
    config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text())
    if not config.environment.docker_image:
        return "no environment.docker_image (OpenSandbox builds no Dockerfile)"
    baseline = config.environment.resolve_baseline()
    if baseline.network_mode == NetworkMode.ALLOWLIST:
        return "an allowlisted network (OpenSandbox enforces public or none)"
    for phase in ("agent", "verifier"):
        policy = getattr(config, phase).explicit_phase_policy()
        if policy is not None and policy != baseline:
            return f"a network of its {phase} phase that is not the environment's (OpenSandbox cannot switch it)"
    return None


def unrunnable_tasks(kwargs: dict[str, Any], tasks: dict[str, Path]) -> dict[str, str]:
    """trial_server.Backend's hook: the tasks why_unrunnable refuses (every one without a server to run on)."""
    if not (kwargs.get("domain") or first_set(ENDPOINT_VARS)):
        return {name: f"no OpenSandbox server: set {' or '.join(ENDPOINT_VARS)}" for name in tasks}
    with ThreadPoolExecutor(32) as pool:
        whys = dict(zip(tasks, pool.map(why_unrunnable, tasks.values()), strict=True))
    return {name: why for name, why in whys.items() if why}


def refuse_unreachable_agent_network(kwargs: dict[str, Any], agent_network: str) -> None:
    """trial_server.Backend's hook: OpenSandbox cannot limit the agent's network (no runtime switch of a sandbox's
    egress): the agent runs with the task's own, said once here rather than refused."""
    if agent_network == "endpoint":
        logger.warning(
            "OpenSandbox: agent_network=endpoint is not enforced: the agent runs with the task's own network "
            "(HARBOR_AGENT_NETWORK=task says so); a harness inside the sandbox reaches a public endpoint only"
        )


def redact(text: str) -> str:
    """``text`` without the server's address (scheme or not) in it."""
    endpoint = first_set(ENDPOINT_VARS)
    if not endpoint:
        return text
    return text.replace(endpoint, "<endpoint>").replace(endpoint_host(endpoint), "<endpoint>")


class RedactingFormatter(logging.Formatter):
    """The CLI's log lines (the SDK's name the server, its retries' URLs), tracebacks included, redacted."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


async def check(image: str) -> str:
    """One sandbox of ``image``, created as OpenSandboxEnvironment creates it, a command's round trip, its kill."""
    from opensandbox import Sandbox

    config, transport = connection_config()
    t0 = time.monotonic()
    try:
        sandbox = await Sandbox.create(
            image,
            timeout=timedelta(minutes=10),
            ready_timeout=timedelta(minutes=5),
            metadata={RUN_LABEL: "check"},
            resource={"cpu": "1", "memory": "1Gi"},
            connection_config=config,
        )
        try:
            execution = await sandbox.commands.run(f"printf {CHECK_OUTPUT!r}")
            output = joined_lines(execution.logs.stdout)
            if output != CHECK_OUTPUT or execution.exit_code != 0:
                raise RuntimeError(f"printf answered {output!r} (exit {execution.exit_code})")
        finally:
            await sandbox.kill()
            await sandbox.close()
    finally:
        if transport is not None:
            await transport.aclose()
    return f"{time.monotonic() - t0:.1f}s"


async def sweep(tag: str, dry_run: bool) -> tuple[int, int]:
    """Kill every live sandbox of the run ``tag`` (its RUN_LABEL); (found, killed)."""
    from opensandbox import SandboxManager
    from opensandbox.models.sandboxes import SandboxFilter

    config, transport = connection_config()
    found = killed = 0
    try:
        manager = await SandboxManager.create(connection_config=config)
        try:
            ids: list[str] = []
            page = 1
            while True:
                paged = await manager.list_sandbox_infos(
                    SandboxFilter(metadata={RUN_LABEL: run_label(tag)}, page_size=100, page=page)
                )
                ids += [i.id for i in paged.sandbox_infos if i.status.state not in ("Terminated", "Stopping")]
                if not paged.pagination.has_next_page:
                    break
                page += 1
            found = len(ids)
            limit = asyncio.Semaphore(SWEEP_CONCURRENCY)

            async def kill(sandbox_id: str) -> bool:
                async with limit:
                    try:
                        await manager.kill_sandbox(sandbox_id)
                        return True
                    except Exception as e:
                        print(f"OPENSANDBOX_SWEEP kill {sandbox_id} failed: {redact(str(e))}", file=sys.stderr)
                        return False

            killed = sum(await asyncio.gather(*(kill(i) for i in ([] if dry_run else ids))))
        finally:
            await manager.close()
    finally:
        if transport is not None:
            await transport.aclose()
    return found, killed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="create a sandbox, run a command in it, kill it")
    check_parser.add_argument("--image", default="ubuntu:24.04")
    sweep_parser = commands.add_parser("sweep", help="kill the live sandboxes of a run")
    sweep_parser.add_argument("--run", required=True, help="its RUN_TAG")
    sweep_parser.add_argument("--dry-run", action="store_true", help="count them, kill none")
    args = parser.parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    if not first_set(ENDPOINT_VARS):
        raise SystemExit(f"FATAL: no OpenSandbox server: set {' or '.join(ENDPOINT_VARS)}")
    if args.command == "check":
        try:
            took = asyncio.run(check(args.image))
        except Exception as e:
            raise SystemExit(f"OPENSANDBOX_CHECK_FAIL {type(e).__name__}: {redact(str(e))[:2000]}") from None
        print(f"OPENSANDBOX_CHECK_OK image={args.image} {took}", flush=True)
        return
    if not run_label(args.run):
        raise SystemExit(f"FATAL: --run {args.run!r} has no label characters")
    try:
        found, killed = asyncio.run(sweep(args.run, args.dry_run))
    except Exception as e:
        raise SystemExit(f"OPENSANDBOX_SWEEP_FAIL {type(e).__name__}: {redact(str(e))[:2000]}") from None
    print(f"OPENSANDBOX_SWEEP run={run_label(args.run)} found={found} killed={killed}", flush=True)


if __name__ == "__main__":
    main()
