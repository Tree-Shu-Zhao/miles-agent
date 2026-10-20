"""Run-to-completion on client disconnect for the stateful gym servers.

nemo-gym 0.7.0 wraps every server in ``ClientDisconnectCancellationMiddleware``
(NVIDIA-NeMo/Gym#3064): when a caller drops the connection, the handler is
cancelled and the cancellation cascades down the agent -> env server chain.
Our servers release env sessions, pool envs, sandbox slots and run records in
best-effort ``/close`` calls and ``except Exception`` cleanup that a
cancellation (a ``BaseException``) skips, so a caller that gives up — the
gym_bridge ``NEMO_GYM_RUN_TIMEOUT`` — would leak them until the TTL sweeper,
or for good. Servers that list this mixin keep the pre-0.7.0 behaviour: a
request runs to completion after its client is gone.
"""

from fastapi import FastAPI


class RunToCompletionOnDisconnect:
    """Skip nemo-gym's disconnect cancellation; list it before the gym base class.

    Overrides ``SimpleServer.setup_cancellation_middleware``, the hook nemo-gym
    calls to install the middleware (tests/test_contracts.py pins that name).
    """

    def setup_cancellation_middleware(self, app: FastAPI) -> None:
        return None
