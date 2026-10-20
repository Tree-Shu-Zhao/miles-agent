"""Vendored-WebShop glue shared by both engine modes (server.py shared-engine
and _engine_worker.py pool). All heavy imports (web_agent_site -> torch/
pyserini/spacy/flask) are confined to build_engine() so this module imports
with stdlib only.

Engine semantics (shared SimServer, fixed-seed goals, per-episode session keys):

- ONE shared ``SimServer`` (catalog + Lucene index + spaCy reward model) built
  once per engine; every episode gets a lightweight
  ``WebAgentTextEnv(server=<shared>)`` rather than a per-episode DB+JVM.
- goal_idx is the SimServer goal index under the goal-shuffle seed (default
  ``GOAL_SHUFFLE_SEED`` = 233, the ORIGINAL WebShop / TCOD constant), so a goal
  index means the same goal everywhere and, on the canonical data (full
  catalog + human goals), the same goal as in the WebShop paper and TCOD's
  train/test files. Seeded BEFORE SimServer: product prices and goal price
  constraints consume the module RNG before SimServer's own seed runs.
- Content identity: a list position is only stable while catalog, goal code
  and seed all stay fixed, so the frozen recipe datasets address goals by
  ``goal_id`` (``agent_utils.webshop_goals``: sha1 of the scoring-relevant
  fields). Each backend builds a ``GoalTable`` over its engine's goal list at
  boot: it resolves ``goal_id -> index`` for /reset, refuses ids the engine
  does not know (``UnknownGoal`` -> HTTP 400, never retried: the dataset and
  the engine disagree on catalog / goal code) and exposes the order-free
  ``goals_digest`` on /health so the training entry can refuse an engine whose
  goal SET differs from the dataset's. ``GET /goals`` (``GoalTable.export``)
  is what ``export_goals.py`` freezes the dataset from.
- observation_mode is the vendored env's rendering: ``text_rich`` (TCOD /
  original WebShop baselines: one visible text per line, ``[button] X
  [button_]`` markup) or ``text`` (verl-agent: ``[SEP]``-joined). The adapter
  must be configured with the same mode (it parses the page); /health
  reports it under ``protocol`` so the training entry can gate on parity.
- Session-key isolation: ``reset(session=goal_idx)`` uses ``str(goal_idx)`` as
  the engine session key, which would collide across concurrent episodes on
  the same goal (GRPO groups!). Each episode gets ``session_prefix='<sid>:'``.
- The vendored env auto-resets onto a random goal on done; the returned obs
  precedes it, and we drop this episode's engine sessions on close so
  ``SimServer.user_sessions`` cannot grow without bound.
- Reserved search markers: upstream ``get_top_n_product_from_keywords`` treats
  a query whose FIRST token is ``<r>``/``<a>``/``<c>``/``<q>`` as an internal
  random/attribute/category/query lookup and crashes (KeyError/IndexError) on
  anything a policy would emit. ``step_env`` neutralizes such a search into the
  upstream no-op action (page unchanged, reward 0) instead of letting it 500.
  The adapter mirrors ``RESERVED_SEARCH_TOKENS`` to count it as invalid.
"""

import logging
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from agent_utils.webshop_goals import goal_id, goal_record, goals_set_digest
except ImportError:
    # The py3.10 engine env cannot pip-install gym_root/agent_utils (it requires
    # py>=3.12 + nemo-gym); webshop_goals itself is stdlib-only, so reach it by
    # path: env_servers/webshop/ and gym_root/ are siblings under the checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gym_root"))
    from agent_utils.webshop_goals import goal_id, goal_record, goals_set_digest

logger = logging.getLogger("webshop_engine")

# Default goal-shuffle seed = the ORIGINAL WebShop constant (web_agent_site/
# app.py and the upstream SimServer both do random.seed(233); random.shuffle
# (goals)); TCOD's princeton-nlp env inherits it, so on the canonical data our
# goal index i is TCOD's task_id i. Overridable per run (--goal-seed), but the
# seed drives the WHOLE build (build_engine seeds the module RNG first):
# product prices, each goal's price_upper / instruction_text (goal.py draws the
# price caps from that RNG before the shuffle) and the list order. A different
# seed is therefore a different goal SET, and the frozen recipe data (which
# addresses goals by goal_id) makes the training entry refuse it via the goals_digest gate
# until the goals file is re-exported under that seed.
GOAL_SHUFFLE_SEED = 233
# Vendored env rendering modes the adapter knows how to parse.
OBSERVATION_MODES = ("text", "text_rich")
DEFAULT_OBSERVATION_MODE = "text_rich"  # TCOD / original WebShop baselines

# First-token markers the upstream search dispatcher reserves (engine.py
# get_top_n_product_from_keywords). Mirrored by the adapter's _format.py.
RESERVED_SEARCH_TOKENS = ("<a>", "<c>", "<q>", "<r>")
# Upstream parse_action's pattern (re.match, greedy): name[arg]. Reused so the
# reserved-token check sees exactly what the engine would see.
_ACTION_RE = re.compile(r"(.+)\[(.+)\]")
# Any action that parses as neither search[...] nor click[<clickable>] takes
# upstream's else-branch: reward 0, done False, page unchanged.
_NOOP_ACTION = "invalid"


@dataclass
class EngineConfig:
    vendored_path: str  # env_servers/webshop/webshop (populated by vendor.sh)
    data_dir: str  # $WEBSHOP_DATA_DIR (items_*.json live here)
    small: bool = True
    human_goals: bool = False
    seed: int = GOAL_SHUFFLE_SEED
    observation_mode: str = DEFAULT_OBSERVATION_MODE
    # Module providing build_engine/make_env/step_env/drop_engine_state,
    # resolved INSIDE each worker process (monkeypatching cannot cross the
    # spawn boundary; tests point this at a fake engine so the pool runs
    # without pyserini/JVM). Default = this module (the real vendored stack).
    factory_module: str = "_webshop_engine"

    def __post_init__(self) -> None:
        if self.observation_mode not in OBSERVATION_MODES:
            raise ValueError(f"observation_mode must be one of {OBSERVATION_MODES}, got {self.observation_mode!r}")

    def file_paths(self) -> tuple[str, str]:
        data = Path(self.data_dir)
        if self.small:
            return str(data / "items_shuffle_1000.json"), str(data / "items_ins_v2_1000.json")
        return str(data / "items_shuffle.json"), str(data / "items_ins_v2.json")

    def protocol(self) -> dict[str, Any]:
        """What defines goal identity and page rendering for this engine —
        surfaced on /health so the training entry can refuse a stale/mismatched
        engine (the adapter and the data both assume these values)."""
        return {
            "seed": self.seed,
            "observation_mode": self.observation_mode,
            "human_goals": bool(self.human_goals),
            "catalog": "small" if self.small else "full",
        }


def goal_index(task_id: Any, n_goals: int | None = None) -> int:
    """Coerce a legacy /reset task_id into a goal index, raising ValueError (->
    HTTP 400, which the adapter does not retry) for anything that is not an int
    or, when ``n_goals`` is known, is out of range."""
    if isinstance(task_id, bool) or task_id is None:
        raise ValueError(f"task_id must be an integer goal index, got {task_id!r}")
    try:
        idx = int(task_id)
    except (TypeError, ValueError):
        raise ValueError(f"task_id must be an integer goal index, got {task_id!r}") from None
    if n_goals is not None and not 0 <= idx < n_goals:
        raise ValueError(f"goal index {idx} out of range [0, {n_goals})")
    return idx


class UnknownGoal(ValueError):
    """/reset named a goal_id the engine's goal list does not contain. A
    ValueError so the HTTP layer answers 400 and the adapter never retries:
    a retry cannot help, the dataset and the engine disagree on the catalog /
    goal code that produced the goals (the job-level digest gate should have
    refused the run; this is the per-episode backstop)."""


class GoalTable:
    """Content identity of one engine's goal list, built once per engine.

    ``ids[i] == goal_id(goals[i])``; ``index`` maps goal_id -> FIRST position
    (should the goal generator ever emit the same content twice, the copies
    score identically, so any one of them serves the id); ``digest`` is the
    order-free set digest surfaced on /health; ``export`` is the /goals payload
    the frozen dataset is built from. Duplicates are logged, not rejected: they
    are a property of the vendored goal code, not of this server.
    """

    def __init__(self, goals: Sequence[Mapping[str, Any]]):
        self._goals = goals
        self.ids: list[str] = [goal_id(g) for g in goals]
        self.index: dict[str, int] = {}
        duplicates = 0
        for i, gid in enumerate(self.ids):
            if gid in self.index:
                duplicates += 1
            else:
                self.index[gid] = i
        if duplicates:
            logger.warning(
                "%d of %d goals duplicate another goal's content; the first position serves each goal_id",
                duplicates,
                len(self.ids),
            )
        self.digest = goals_set_digest(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def unique(self) -> int:
        """Distinct questions (== len(self) unless the generator repeated content)."""
        return len(self.index)

    def resolve(self, goal_ref: Any) -> int:
        """Goal list index for a /reset reference: a str goal_id (content
        address) or a legacy int position (range-checked by ``goal_index``)."""
        if isinstance(goal_ref, str):
            idx = self.index.get(goal_ref)
            if idx is None:
                raise UnknownGoal(
                    f"goal_id {goal_ref!r} is not in the engine goal list (goals_digest {self.digest}): "
                    "the dataset and the engine disagree on catalog / goal code"
                )
            return idx
        return goal_index(goal_ref, len(self.ids))

    def export(self) -> list[dict[str, Any]]:
        """Every goal in list order as ``{"goal_id", "goal_idx", **goal_record}``
        (JSON-safe; ``weight`` and other non-identity fields are dropped)."""
        return [
            {"goal_id": gid, "goal_idx": i, **goal_record(g)}
            for i, (gid, g) in enumerate(zip(self.ids, self._goals, strict=True))
        ]


def is_reserved_search(action: str) -> bool:
    """True when ``action`` is a search whose first query token (upstream
    tokenization: lowercase, split on single spaces) is a reserved marker."""
    m = _ACTION_RE.match(action)
    if m is None or m.group(1) != "search":
        return False
    return m.group(2).lower().split(" ")[0] in RESERVED_SEARCH_TOKENS


def build_engine(cfg: EngineConfig):
    """Build the shared SimServer (heavy: catalog load + Lucene + spaCy + JVM).
    Returns the engine and the WebAgentTextEnv class (so callers make light
    per-episode envs without re-importing)."""
    vendored = Path(cfg.vendored_path)
    if not (vendored / "web_agent_site").is_dir():
        raise SystemExit(
            f"vendored webshop tree not found at {vendored}; run "
            "env_servers/webshop/vendor.sh then env_servers/webshop/setup.sh"
        )
    sys.path.insert(0, str(vendored))
    from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv

    file_path, attr_path = cfg.file_paths()
    if not Path(file_path).is_file():
        raise SystemExit(
            f"{file_path} missing — run setup.sh {'' if cfg.small else '--full'} first "
            "(and make sure its catalog mode matches --small)"
        )

    # Seed BEFORE SimServer: prices + goal price constraints consume the module
    # RNG before SimServer's own random.seed(seed), so without this goal_idx ->
    # price constraint is not reproducible (same ordering as WebAgentTextEnv).
    import random

    random.seed(cfg.seed)
    engine = SimServer(
        cfg.seed,
        "http://127.0.0.1:3000",
        file_path,
        attr_path,
        filter_goals=None,
        limit_goals=-1,
        num_products=None,  # whole configured file, via the default 'indexes' Lucene dir
        human_goals=cfg.human_goals,
        show_attrs=False,
    )
    return engine, WebAgentTextEnv, (file_path, attr_path)


def make_env(
    engine,
    env_cls,
    file_paths: tuple[str, str],
    goal_idx: int,
    prefix: str,
    observation_mode: str = DEFAULT_OBSERVATION_MODE,
) -> tuple[Any, str, dict]:
    """Lightweight per-episode env sharing ``engine``. On failure, prune whatever
    the half-built env wrote under this prefix (it is not registered yet)."""
    if not 0 <= goal_idx < len(engine.goals):
        raise ValueError(f"goal index {goal_idx} out of range [0, {len(engine.goals)})")
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(f"observation_mode must be one of {OBSERVATION_MODES}, got {observation_mode!r}")
    file_path, attr_path = file_paths
    try:
        env = env_cls(
            observation_mode=observation_mode,
            file_path=file_path,
            attr_path=attr_path,
            server=engine,
            session_prefix=prefix,
        )
        obs, _ = env.reset(session=goal_idx)
        avail = env.get_available_actions()
    except Exception:
        drop_engine_state(engine, prefix)
        raise
    return env, obs, {"available_actions": avail, "task_score": 0.0, "won": False, "goal_idx": goal_idx}


def step_env(engine, env, action: str) -> tuple[str, float, bool, dict]:
    """One WebShop step. Reward is the RAW score at the terminal buy step
    (info.task_score); the remap to the training reward lives in
    recipes/webshop/reward.py, so the engine stays neutral."""
    if is_reserved_search(action):
        # Would crash inside upstream's keyword dispatcher; make it the same
        # no-op an unparseable action already is (page unchanged, reward 0).
        action = _NOOP_ACTION
    obs, raw_reward, done, _ = env.step(action)
    raw = float(raw_reward)
    won = bool(done) and raw == 1.0
    if done:
        # The vendored env auto-resets onto a random NEW goal on done, so
        # env.get_available_actions() here would advertise that unrelated goal's
        # search page. A finished episode has no next action; return an empty
        # admissible set (the terminal obs is model-visible but never acted on).
        drop_engine_state(engine, getattr(env, "session_prefix", None))
        avail: dict = {"has_search_bar": False, "clickables": []}
    else:
        avail = env.get_available_actions()
    return obs, raw, bool(done), {"available_actions": avail, "task_score": raw, "won": won}


def drop_engine_state(engine, prefix: str | None) -> None:
    """Prune this episode's SimServer sessions (incl. any auto-reset junk).
    Caller serializes access to the engine (worker single-thread or RLock)."""
    if not prefix:
        return
    stale = [k for k in engine.user_sessions if k.startswith(prefix)]
    for k in stale:
        del engine.user_sessions[k]
