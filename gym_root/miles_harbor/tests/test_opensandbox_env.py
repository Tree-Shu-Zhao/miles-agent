"""miles_harbor.opensandbox_env: Harbor's OpenSandboxEnvironment with its server from the environment, against a fake
of the SDK's Sandbox (its commands answer as execd streams: a message a line, the terminator stripped, an empty line
"\\n") and SandboxManager -- no server, no network -- and the trial server's hooks on task dirs."""

import asyncio
import logging
import re
import ssl
import sys
from datetime import timedelta
from http import HTTPStatus
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")  # the Harbor package is in gym_root/miles_harbor's venv, not in every checkout
pytest.importorskip("opensandbox")  # harbor[opensandbox]

import certifi  # noqa: E402
import httpx  # noqa: E402
import miles_harbor.opensandbox_env as ose  # noqa: E402
import opensandbox  # noqa: E402
from harbor.models.task.config import EnvironmentConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402
from miles_harbor import trial_server  # noqa: E402
from miles_harbor.opensandbox_env import (  # noqa: E402
    RUN_LABEL,
    SANDBOX_TIMEOUT_S,
    OpenSandboxEnvironment,
    joined_lines,
    redact,
    refuse_unreachable_agent_network,
    run_label,
    unrunnable_tasks,
)
from opensandbox.transport import RetryAsyncTransport, RetryPolicy, unwrap_retry_transport  # noqa: E402

ENDPOINT = "https://sandbox.example.test:8443"
SERVER_VARS = ("OPENSANDBOX_ENDPOINT", "OPENSANDBOX_DOMAIN", "OPENSANDBOX_KEY", "OPENSANDBOX_API_KEY")


@pytest.fixture
def server(monkeypatch):
    """The server as a job's environment names it; none of the other knobs."""
    for name in (*SERVER_VARS, "OPENSANDBOX_CA_FILE", "OPENSANDBOX_PROXY", "RUN_TAG", "SSL_CERT_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENSANDBOX_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("OPENSANDBOX_KEY", "k-test")
    monkeypatch.setattr(logging.getLogger("httpx"), "filters", [])  # an environment adds one: the test's alone


def make_env(tmp_path, **kwargs):
    """The environment as Harbor's factory builds it for a task with a prebuilt image."""
    (tmp_path / "environment").mkdir(exist_ok=True)
    return OpenSandboxEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="t1",
        session_id="s1",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="org/img:1"),
        **kwargs,
    )


def execution(stdout=(), stderr=(), exit_code=0, error=None):
    """A foreground command's result as the SDK returns it: execd's messages, its exit code, its error."""

    def messages(texts):
        return [SimpleNamespace(text=text) for text in texts]

    return SimpleNamespace(
        logs=SimpleNamespace(stdout=messages(stdout), stderr=messages(stderr)), exit_code=exit_code, error=error
    )


class FakeSandbox:
    """The SDK's Sandbox: create records its arguments; each command answers the next of ``answers`` (an execution,
    or a coroutine function giving one; a new sandbox's: ``script``'s), else an empty success."""

    created: list["FakeSandbox"] = []
    script: list = []

    def __init__(self, image, kwargs):
        self.image, self.kwargs = image, kwargs
        self.answers: list = list(self.script)
        self.ran: list = []
        self.killed = self.closed = False
        self.commands = SimpleNamespace(run=self.run)
        self.files = SimpleNamespace(create_directories=self.nothing, write_files=self.nothing)

    @classmethod
    async def create(cls, image, **kwargs):
        sandbox = cls(image, kwargs)
        cls.created.append(sandbox)
        return sandbox

    async def get_info(self):
        return SimpleNamespace(status=SimpleNamespace(state="Running"))

    async def is_healthy(self):
        return True

    async def nothing(self, entries):
        return None

    async def run(self, command, opts=None):
        self.ran.append((command, opts))
        answer = self.answers.pop(0) if self.answers else execution()
        return await answer() if callable(answer) else answer

    async def kill(self):
        self.killed = True

    async def close(self):
        self.closed = True


@pytest.fixture
def sandboxes(monkeypatch):
    """The SDK's Sandbox faked (Harbor's environment imports it as it starts); the sandboxes created."""
    monkeypatch.setattr(opensandbox, "Sandbox", FakeSandbox)
    FakeSandbox.created, FakeSandbox.script = [], []
    return FakeSandbox.created


class RecordingTransport(httpx.AsyncHTTPTransport):
    """httpx's transport, its arguments and its close recorded."""

    made: list["RecordingTransport"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs, self.closed = kwargs, False
        RecordingTransport.made.append(self)

    async def aclose(self):
        self.closed = True
        await super().aclose()


@pytest.fixture
def transports(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", RecordingTransport)
    RecordingTransport.made = []
    return RecordingTransport.made


def test_execds_messages_are_joined_into_the_lines_printed():
    # printf 'a\n\nb' and printf 'a\n\nb\n' alike: the last line's newline is not known, it always gets one
    assert joined_lines(execution(stdout=["a", "\n", "b"]).logs.stdout) == "a\n\nb\n"
    assert joined_lines([]) == ""
    # a message that kept its terminator keeps it, once
    assert joined_lines(execution(stdout=["x\n", "y"]).logs.stdout) == "x\ny\n"


def test_the_server_and_its_key_are_the_environments_never_the_kwargs(tmp_path, server, monkeypatch):
    env = make_env(tmp_path)
    assert (env._domain, env._api_key, env._protocol, env._use_server_proxy) == (ENDPOINT, "k-test", "https", True)
    assert env._sandbox_timeout_sec == SANDBOX_TIMEOUT_S and env._metadata == {}
    # Harbor's names when ours are unset; ours first when both are
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "https://harbor.example.test")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "k-harbor")
    assert (make_env(tmp_path)._domain, make_env(tmp_path)._api_key) == (ENDPOINT, "k-test")
    monkeypatch.delenv("OPENSANDBOX_ENDPOINT")
    monkeypatch.delenv("OPENSANDBOX_KEY")
    assert (make_env(tmp_path)._domain, make_env(tmp_path)._api_key) == ("https://harbor.example.test", "k-harbor")
    # a domain kwarg is taken; a key in the kwargs is refused (the job's config is printed); no server at all is
    assert make_env(tmp_path, domain="https://kw.example.test")._domain == "https://kw.example.test"
    with pytest.raises(ValueError, match="no api_key kwarg"):
        make_env(tmp_path, api_key="k-kwarg")
    monkeypatch.delenv("OPENSANDBOX_DOMAIN")
    with pytest.raises(ValueError, match="no server: set OPENSANDBOX_ENDPOINT or OPENSANDBOX_DOMAIN"):
        make_env(tmp_path)


def test_every_connection_takes_the_endpoints_scheme(tmp_path, server, monkeypatch):
    # the SDK takes the execd plane's (the key travels there too) from protocol alone: it follows the endpoint's
    for endpoint, protocol in (("http://h.test", "http"), ("HTTPS://h.test", "https"), ("h.test:8080", "https")):
        monkeypatch.setenv("OPENSANDBOX_ENDPOINT", endpoint)
        assert make_env(tmp_path)._protocol == protocol
        assert ose.connection_config()[0].protocol == protocol
    assert make_env(tmp_path, protocol="http")._protocol == "http"  # no scheme: the kwarg's
    monkeypatch.setenv("OPENSANDBOX_ENDPOINT", "https://h.test")
    assert make_env(tmp_path, protocol="http")._protocol == "https"


def test_a_ca_file_and_a_proxy_are_one_transport_of_the_environments_own(tmp_path, server, monkeypatch, transports):
    sdk = make_env(tmp_path)._load_opensandbox()
    # neither: the SDK's own transport (it builds and closes it)
    config = make_env(tmp_path)._build_connection_config(sdk)
    assert config.transport is None and config.use_server_proxy and config.protocol == "https"
    assert (config.domain, config.api_key) == (ENDPOINT, "k-test") and transports == []
    # the CA trusted besides SSL_CERT_FILE's (else certifi's); the proxy; one transport for every create attempt,
    # the SDK told it is not its own (a sandbox's close leaves it open), its create-latency report off
    pem = open(certifi.where()).read()
    certs = re.findall(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----\n", pem, re.S)
    (tmp_path / "bundle.pem").write_text(certs[0])
    (tmp_path / "server.pem").write_text(certs[1])
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "bundle.pem"))
    monkeypatch.setenv("OPENSANDBOX_CA_FILE", str(tmp_path / "server.pem"))
    monkeypatch.setenv("OPENSANDBOX_PROXY", "http://proxy.example.test:3128")
    env = make_env(tmp_path)
    config = env._build_connection_config(sdk)
    assert env._build_connection_config(sdk).transport is config.transport and len(transports) == 1
    # under the SDK's retry wrapper, httpx's transport behind the proxy's refusals told as connect failures
    pinned = unwrap_retry_transport(config.transport)
    assert isinstance(pinned, ose.ProxyConnectFailures) and pinned.inner is transports[0]
    assert not config.owns_transport and config.disable_metrics
    assert (config.domain, config.api_key, config.protocol) == (
        ENDPOINT,
        "k-test",
        "https",
    ) and config.use_server_proxy
    verify = transports[0].kwargs["verify"]
    assert isinstance(verify, ssl.SSLContext) and len(verify.get_ca_certs()) == 2
    assert transports[0].kwargs["proxy"] == "http://proxy.example.test:3128"
    # the kwargs over the knobs
    make_env(tmp_path, ca_file=str(tmp_path / "bundle.pem"), proxy="http://p2.test:1")._build_connection_config(sdk)
    assert transports[1].kwargs["proxy"] == "http://p2.test:1"
    assert len(transports[1].kwargs["verify"].get_ca_certs()) == 1
    # stop() closes it, a sandbox or not
    asyncio.run(env.stop(delete=True))
    assert transports[0].closed and env._transport is None
    # a CA file that is not there is an error naming it, not a connection without it
    monkeypatch.setenv("OPENSANDBOX_CA_FILE", str(tmp_path / "missing.pem"))
    with pytest.raises(ValueError, match=r"no CA file .*missing\.pem \(OPENSANDBOX_CA_FILE"):
        make_env(tmp_path)._build_connection_config(sdk)
    with pytest.raises(ValueError, match="no CA file"):
        ose.connection_config()


@pytest.fixture
def retries(monkeypatch):
    """A RetryLog of the test's own."""
    log = ose.RetryLog()
    monkeypatch.setattr(ose, "RETRIES", log)
    return log


async def test_a_post_answered_429_is_retried_after_its_retry_after(retries, caplog):
    # the server defers a create it rate-limits: 429 + Retry-After, nothing created; the SDK alone would not retry it
    assert HTTPStatus.TOO_MANY_REQUESTS not in RetryPolicy().retryable_statuses_for("POST")
    policy = ose.retry_policy(RetryPolicy())
    assert HTTPStatus.TOO_MANY_REQUESTS in policy.retryable_statuses_for("POST") and policy.max_retries == 3
    statuses = [429, 429, 200]
    seen = []

    def server(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(statuses[len(seen) - 1], headers={"Retry-After": "0"})

    transport = RetryAsyncTransport(httpx.MockTransport(server), policy, owns_inner=True)
    with caplog.at_level(logging.WARNING, logger=ose.__name__):
        async with httpx.AsyncClient(transport=transport) as client:
            answer = await client.post("https://sandbox.example.test:8443/v1/sandboxes", json={"image": "x"})
    assert answer.status_code == 200 and seen == [("POST", "/v1/sandboxes")] * 3
    assert retries.total == {"status_429": 2}
    # a burst is one line: the cause's first retry, its count since the last line, the path but never the host
    assert [m for m in caplog.messages if m.startswith("OPENSANDBOX_RETRY")] == [
        "OPENSANDBOX_RETRY cause=status_429 n=1 total=1 backoff_s=0.0 POST /v1/sandboxes"
    ]


def test_a_causes_retries_are_a_line_per_interval(retries, caplog, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(ose.time, "monotonic", lambda: now[0])
    event = SimpleNamespace(
        cause=SimpleNamespace(value="status_429"), backoff=timedelta(seconds=2), method="POST", url=ENDPOINT + "/v1/x"
    )
    with caplog.at_level(logging.WARNING, logger=ose.__name__):
        for _ in range(5):
            retries.note(event)
        now[0] += ose.RETRY_LOG_S
        retries.note(event)
        retries.note(SimpleNamespace(**{**vars(event), "cause": SimpleNamespace(value="status_503")}))
    assert [m for m in caplog.messages if m.startswith("OPENSANDBOX_RETRY")] == [
        "OPENSANDBOX_RETRY cause=status_429 n=1 total=1 backoff_s=2.0 POST /v1/x",
        "OPENSANDBOX_RETRY cause=status_429 n=5 total=6 backoff_s=2.0 POST /v1/x",
        "OPENSANDBOX_RETRY cause=status_503 n=1 total=1 backoff_s=2.0 POST /v1/x",
    ]
    assert retries.total == {"status_429": 6, "status_503": 1}


def test_every_connection_retries_a_post_answered_429(tmp_path, server, monkeypatch, transports, retries):
    sdk = make_env(tmp_path)._load_opensandbox()
    policies = []
    pinned = ose.pinned_transport

    def recording(policy, *args):
        policies.append(policy)
        return pinned(policy, *args)

    monkeypatch.setattr(ose, "pinned_transport", recording)
    # the SDK's own transport (no CA file, no proxy), then the pinned one, for the environment and the CLI alike
    configs = [make_env(tmp_path)._build_connection_config(sdk), ose.connection_config()[0]]
    monkeypatch.setenv("OPENSANDBOX_CA_FILE", certifi.where())
    configs += [make_env(tmp_path)._build_connection_config(sdk), ose.connection_config()[0]]
    for config in configs:
        assert HTTPStatus.TOO_MANY_REQUESTS in config.retry_policy.retryable_statuses_for("POST")
    assert configs[0].transport is None and configs[1].transport is None
    assert policies == [configs[2].retry_policy, configs[3].retry_policy]
    assert isinstance(configs[2].transport, RetryAsyncTransport)


async def test_creates_in_flight_are_bounded_per_process(tmp_path, server, monkeypatch):
    inside, peak = 0, 0

    async def create(self, sdk):
        nonlocal inside, peak
        inside += 1
        peak = max(peak, inside)
        await asyncio.sleep(0.01)
        inside -= 1
        return "sandbox"

    monkeypatch.setattr(ose.HarborOpenSandboxEnvironment, "_create_sandbox", create)
    envs = [make_env(tmp_path) for _ in range(10)]
    monkeypatch.setenv("OPENSANDBOX_MAX_CREATES", "3")
    assert await asyncio.gather(*(env._create_sandbox({}) for env in envs)) == ["sandbox"] * 10
    assert peak == 3
    # 0: no bound
    peak = 0
    monkeypatch.setenv("OPENSANDBOX_MAX_CREATES", "0")
    await asyncio.gather(*(env._create_sandbox({}) for env in envs))
    assert peak == 10


async def test_a_sandbox_is_created_with_the_runs_label_and_the_servers_ttl(
    tmp_path, server, monkeypatch, sandboxes, transports
):
    monkeypatch.setenv("RUN_TAG", "tmax run/7")
    monkeypatch.setenv("OPENSANDBOX_CA_FILE", certifi.where())
    env = make_env(tmp_path, metadata={"team": "x"})
    await env.start(force_build=False)
    (sandbox,) = sandboxes
    assert sandbox.image == "org/img:1" and sandbox.kwargs["timeout"] == timedelta(seconds=SANDBOX_TIMEOUT_S)
    assert sandbox.kwargs["metadata"] == {RUN_LABEL: "tmax-run-7", "team": "x", "session_id": "s1"}
    config = sandbox.kwargs["connection_config"]
    assert (config.domain, config.protocol, config.use_server_proxy) == (ENDPOINT, "https", True)
    assert unwrap_retry_transport(config.transport) is transports[0]
    assert sandbox.ran[0][0].startswith("chmod 777 ")  # Harbor's start, through this exec
    await env.stop(delete=True)
    assert sandbox.killed and sandbox.closed and transports[0].closed
    # a metadata kwarg of the label's key is the label
    assert make_env(tmp_path, metadata={RUN_LABEL: "mine"})._metadata == {RUN_LABEL: "mine"}


async def test_exec_answers_the_output_as_printed_and_a_timeout_as_dockers(tmp_path, server, monkeypatch, sandboxes):
    env = make_env(tmp_path)
    with pytest.raises(RuntimeError, match="Sandbox not found"):
        await env.exec("true")
    await env.start(force_build=False)
    (sandbox,) = sandboxes
    sandbox.answers = [
        execution(stdout=["a", "\n", "b"], stderr=["warn"], exit_code=3),
        execution(exit_code=None),
        execution(exit_code=None, error=SimpleNamespace(name="E", value="x")),
        execution(exit_code=-1, error=SimpleNamespace(name="Killed", value="signal: killed")),
    ]
    result = await env.exec("printf 'a\\n\\nb'", cwd="/app", env={"X": "1"}, timeout_sec=60)
    assert (result.stdout, result.stderr, result.return_code) == ("a\n\nb\n", "warn\n", 3)
    command, opts = sandbox.ran[-1]
    assert command == "printf 'a\\n\\nb'" and opts.working_directory == "/app" and opts.envs == {"X": "1"}
    assert opts.timeout == timedelta(seconds=60)
    assert (await env.exec("true")).return_code == 0  # no exit code, no error
    assert (await env.exec("false")).return_code == 1  # no exit code, an error
    assert (await env.exec("kill -9 $$", timeout_sec=60)).return_code == -1  # killed before its timeout: as it is

    # killed at its timeout (execd: -1): Harbor's docker environment's error
    async def killed_at_the_timeout():
        await asyncio.sleep(1.05)
        return execution(stdout=["partial"], exit_code=-1)

    sandbox.answers = [killed_at_the_timeout]
    with pytest.raises(RuntimeError, match=r"^Command timed out after 1 seconds$"):
        await env.exec("sleep 9", timeout_sec=1)

    # the stream silent past the timeout and its grace: the same
    async def silent():
        await asyncio.sleep(3600)

    monkeypatch.setattr(ose, "STREAM_GRACE_S", 0)
    sandbox.answers = [silent]
    with pytest.raises(RuntimeError, match=r"^Command timed out after 1 seconds$"):
        await env.exec("sleep 9", timeout_sec=1)


async def test_a_proxys_refused_connect_is_retried_as_a_connect_failure(retries):
    # the SDK retries a connect failure on any method (nothing left the client) but a ProxyError on idempotent
    # methods alone; a proxy that cannot reach the server refuses the CONNECT, which is the former
    policy = ose.retry_policy(RetryPolicy(initial_backoff=timedelta(0), max_backoff=timedelta(0)))
    calls = []

    def server(request):
        calls.append(request.method)
        if len(calls) < 3:
            raise httpx.ProxyError("CONNECT answered 502")
        return httpx.Response(200)

    bare = RetryAsyncTransport(httpx.MockTransport(server), policy, owns_inner=True)
    async with httpx.AsyncClient(transport=bare) as client:
        with pytest.raises(httpx.ProxyError):
            await client.post("https://sandbox.example.test:8443/v1/sandboxes", json={})
    assert calls == ["POST"] and retries.total == {}
    calls.clear()
    pinned = RetryAsyncTransport(ose.ProxyConnectFailures(httpx.MockTransport(server)), policy, owns_inner=True)
    async with httpx.AsyncClient(transport=pinned) as client:
        answer = await client.post("https://sandbox.example.test:8443/v1/sandboxes", json={})
    assert answer.status_code == 200 and calls == ["POST"] * 3 and retries.total == {"pre_send": 2}
    # past the policy's retries the failure is told as the connect failure it was taken for, the refusal its cause
    calls.clear()
    policy = ose.retry_policy(RetryPolicy(max_retries=1, initial_backoff=timedelta(0), max_backoff=timedelta(0)))
    pinned = RetryAsyncTransport(ose.ProxyConnectFailures(httpx.MockTransport(server)), policy, owns_inner=True)
    async with httpx.AsyncClient(transport=pinned) as client:
        with pytest.raises(httpx.ConnectError, match="the proxy did not connect: CONNECT answered 502") as failed:
            await client.post("https://sandbox.example.test:8443/v1/sandboxes", json={})
    assert isinstance(failed.value.__cause__, httpx.ProxyError) and calls == ["POST"] * 2


async def test_exec_retries_a_command_whose_connection_failed_before_it_ran(
    tmp_path, server, monkeypatch, sandboxes, retries
):
    from opensandbox.exceptions import SandboxConnectionException

    env = make_env(tmp_path)
    await env.start(force_build=False)  # runs Harbor's setup commands in the sandbox
    (sandbox,) = sandboxes
    ran = len(sandbox.ran)
    monkeypatch.setattr(ose, "EXEC_RETRY_BACKOFF_S", (0.0,))

    def failing(error):
        async def answer():
            raise SandboxConnectionException("the stream failed", cause=error)

        return answer

    # the SDK keeps httpx's error as the cause: the proxy's CONNECT refused, then the TCP connect, then it ran
    sandbox.answers = [failing(httpx.ProxyError("CONNECT 502")), failing(httpx.ConnectError("refused")), execution()]
    assert (await env.exec("echo ok", timeout_sec=60)).return_code == 0
    assert [command for command, _ in sandbox.ran[ran:]] == ["echo ok"] * 3
    assert retries.total == {"exec_pre_send_ProxyError": 1, "exec_pre_send_ConnectError": 1}
    # a failure once the command may have run (its stream broke) is not retried
    sandbox.answers = [failing(httpx.ReadError("reset")), execution()]
    with pytest.raises(SandboxConnectionException, match="the stream failed"):
        await env.exec("echo ok")
    assert len(sandbox.ran) == ran + 4
    # nor one past the budget (OPENSANDBOX_EXEC_RETRY_S; 0: never)
    monkeypatch.setenv("OPENSANDBOX_EXEC_RETRY_S", "0")
    sandbox.answers = [failing(httpx.ConnectError("refused")), execution()]
    with pytest.raises(SandboxConnectionException):
        await env.exec("echo ok")
    assert len(sandbox.ran) == ran + 5 and retries.total["exec_pre_send_ConnectError"] == 1


def test_httpxs_line_per_request_to_the_server_is_dropped(tmp_path, server, caplog):
    make_env(tmp_path)
    make_env(tmp_path)  # one filter per host
    httpx_logger = logging.getLogger("httpx")
    assert [f.host for f in httpx_logger.filters] == ["sandbox.example.test:8443"]
    line = 'HTTP Request: %s %s "%s %d %s"'
    with caplog.at_level(logging.INFO, logger="httpx"):
        httpx_logger.info(line, "POST", f"{ENDPOINT}/v1/sandboxes/s/proxy/44772/command", "HTTP/1.1", 200, "OK")
        httpx_logger.info(line, "POST", "http://127.0.0.1:7100/v1/chat/completions", "HTTP/1.1", 200, "OK")
        httpx_logger.warning("retrying %s/v1/sandboxes", ENDPOINT)
    assert [r.getMessage() for r in caplog.records] == [
        'HTTP Request: POST http://127.0.0.1:7100/v1/chat/completions "HTTP/1.1 200 OK"',
        f"retrying {ENDPOINT}/v1/sandboxes",
    ]


def test_a_run_tag_is_a_label_value():
    assert run_label("os-oracle-1") == "os-oracle-1" and run_label("tmax run/7") == "tmax-run-7"
    assert run_label("x" * 70) == "x" * 63 and run_label("_a.") == "a" and run_label("//") == ""


def write_task(root, toml, *, compose=False):
    (root / "environment").mkdir(parents=True)
    (root / "task.toml").write_text(toml)
    if compose:
        (root / "environment" / "docker-compose.yaml").write_text("services: {}\n")
    return root


def test_unrunnable_tasks_are_those_harbor_would_refuse_on_opensandbox(tmp_path, server, monkeypatch):
    image = '[environment]\ndocker_image = "org/img"\n'
    tasks = {
        "ok": write_task(tmp_path / "ok", image),
        "ok-none": write_task(tmp_path / "ok-none", image + 'network_mode = "no-network"\n'),
        "ok-verifier": write_task(tmp_path / "ok-verifier", image + '[verifier]\nnetwork_mode = "public"\n'),
        "build": write_task(tmp_path / "build", "[environment]\n"),
        "compose": write_task(tmp_path / "compose", image, compose=True),
        "allowlist": write_task(
            tmp_path / "allowlist", image + 'network_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n'
        ),
        "agent-off": write_task(tmp_path / "agent-off", image + '[agent]\nnetwork_mode = "no-network"\n'),
    }
    whys = unrunnable_tasks({}, tasks)
    assert sorted(whys) == ["agent-off", "allowlist", "build", "compose"]
    assert "no environment.docker_image" in whys["build"] and "docker-compose" in whys["compose"]
    assert "allowlisted" in whys["allowlist"] and "agent phase" in whys["agent-off"]
    # no server: every task, said so; a domain kwarg is one
    monkeypatch.delenv("OPENSANDBOX_ENDPOINT")
    assert set(unrunnable_tasks({}, {"ok": tasks["ok"]}).values()) == {
        "no OpenSandbox server: set OPENSANDBOX_ENDPOINT or OPENSANDBOX_DOMAIN"
    }
    assert unrunnable_tasks({"domain": "https://kw.test"}, {"ok": tasks["ok"]}) == {}


def test_the_agents_network_is_the_tasks_said_once(caplog):
    with caplog.at_level(logging.WARNING, logger=ose.__name__):
        refuse_unreachable_agent_network({}, "task")
        assert caplog.records == []
        refuse_unreachable_agent_network({}, "endpoint")  # a warning, not a SystemExit
    assert "agent_network=endpoint is not enforced" in caplog.text


def test_the_trial_server_finds_the_hooks_on_the_backends_module(tmp_path, server):
    backend = trial_server.load_backend("miles_harbor.opensandbox_env:OpenSandboxEnvironment")
    write_task(tmp_path / "build", "[environment]\n")
    assert list(backend.unrunnable_tasks({}, {"build": tmp_path / "build"})) == ["build"]
    task_config = SimpleNamespace(agent=None)
    backend.limit_agent_network(task_config, None)  # the default: the task's config as it is
    assert task_config == SimpleNamespace(agent=None)


def test_the_cli_never_prints_the_endpoint(server, monkeypatch, capsys):
    host = "sandbox.example.test:8443"
    assert redact(f"POST {ENDPOINT}/v1 and {host}") == "POST <endpoint>/v1 and <endpoint>"
    formatter = ose.RedactingFormatter("%(message)s")
    record = logging.LogRecord("x", logging.WARNING, __file__, 1, "retrying %s", (f"{ENDPOINT}/v1",), None)
    assert formatter.format(record) == "retrying <endpoint>/v1"

    def main(*argv):
        monkeypatch.setattr(sys, "argv", ["opensandbox_env", *argv])
        ose.main()

    async def unreachable(image):
        raise ConnectionError(f"cannot reach {ENDPOINT}/v1/sandboxes ({host})")

    monkeypatch.setattr(ose, "check", unreachable)
    with pytest.raises(SystemExit) as exit_:
        main("check")
    assert str(exit_.value) == (
        "OPENSANDBOX_CHECK_FAIL ConnectionError: cannot reach <endpoint>/v1/sandboxes (<endpoint>)"
    )

    async def checked(image):
        return f"{image} 1.0s"

    async def swept(tag, dry_run):
        return (2, 0 if dry_run else 2)

    monkeypatch.setattr(ose, "check", checked)
    monkeypatch.setattr(ose, "sweep", swept)
    main("check", "--image", "alpine:3")
    main("sweep", "--run", "tmax run/7", "--dry-run")
    assert capsys.readouterr().out.splitlines() == [
        "OPENSANDBOX_CHECK_OK image=alpine:3 alpine:3 1.0s",
        "OPENSANDBOX_SWEEP run=tmax-run-7 found=2 killed=0",
    ]
    with pytest.raises(SystemExit, match="has no label characters"):
        main("sweep", "--run", "//")
    monkeypatch.delenv("OPENSANDBOX_ENDPOINT")
    with pytest.raises(SystemExit, match="FATAL: no OpenSandbox server"):
        main("check")


async def test_check_round_trips_a_command_and_kills_its_sandbox(server, sandboxes):
    FakeSandbox.script = [execution(stdout=["a", "\n", "b"])]
    assert (await ose.check("ubuntu:24.04")).endswith("s")
    (sandbox,) = sandboxes
    assert sandbox.ran[0][0] == "printf 'a\\n\\nb\\n'" and sandbox.kwargs["metadata"] == {RUN_LABEL: "check"}
    assert sandbox.kwargs["connection_config"].protocol == "https" and sandbox.killed and sandbox.closed
    # an answer that lost its empty line fails, the sandbox killed all the same
    FakeSandbox.script = [execution(stdout=["a", "b"])]
    with pytest.raises(RuntimeError, match="printf answered 'a\\\\nb\\\\n'"):
        await ose.check("ubuntu:24.04")
    assert sandboxes[1].killed


class FakeManager:
    """The SDK's SandboxManager: two pages of a run's sandboxes, one already terminated; kills recorded."""

    filters: list = []
    killed: list = []
    pages: dict = {}
    in_flight = most = 0

    @classmethod
    async def create(cls, connection_config):
        return cls()

    async def list_sandbox_infos(self, filter):
        FakeManager.filters.append(filter)
        infos = [SimpleNamespace(id=i, status=SimpleNamespace(state=s)) for i, s in FakeManager.pages[filter.page]]
        return SimpleNamespace(
            sandbox_infos=infos, pagination=SimpleNamespace(has_next_page=filter.page < len(FakeManager.pages))
        )

    async def kill_sandbox(self, sandbox_id):
        FakeManager.in_flight += 1
        FakeManager.most = max(FakeManager.most, FakeManager.in_flight)
        await asyncio.sleep(0.01)
        FakeManager.in_flight -= 1
        if sandbox_id == "bad":
            raise RuntimeError(f"{ENDPOINT}: 500")
        FakeManager.killed.append(sandbox_id)

    async def close(self):
        return None


async def test_sweep_kills_the_live_sandboxes_of_a_run(server, monkeypatch, capsys):
    monkeypatch.setattr(opensandbox, "SandboxManager", FakeManager)
    FakeManager.pages = {1: [("a", "Running"), ("b", "Terminated")], 2: [("c", "Pending"), ("bad", "Paused")]}
    FakeManager.filters, FakeManager.killed = [], []
    assert await ose.sweep("tmax run/7", dry_run=True) == (3, 0) and FakeManager.killed == []
    assert [(f.metadata, f.page) for f in FakeManager.filters] == [({RUN_LABEL: "tmax-run-7"}, p) for p in (1, 2)]
    assert await ose.sweep("tmax run/7", dry_run=False) == (3, 2) and sorted(FakeManager.killed) == ["a", "c"]
    assert "kill bad failed: <endpoint>: 500" in capsys.readouterr().err
    # a big run's sandboxes are killed SWEEP_CONCURRENCY at a time, not one after another
    FakeManager.pages = {1: [(f"s{n}", "Running") for n in range(40)]}
    FakeManager.killed, FakeManager.most = [], 0
    assert await ose.sweep("big", dry_run=False) == (40, 40) and FakeManager.most == ose.SWEEP_CONCURRENCY
