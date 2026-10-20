"""TextWorld hardening for py3.13 + one multi-session server process.

Validated against alfworld 0.4.2 / textworld 1.7.0 / tatsu 5.8.3 (spike
probes, 2026-08-31); every patch here reproduces a concrete crash when
removed:

1. PEP 667: ``EvalSymbol.derive`` writes grammar variables into ``locals()``
   and expects a bare ``eval()`` to see them; on py3.13 ``locals()`` is an
   independent snapshot, so every PDDL feedback derivation dies with
   ``NameError``.
2. textworld holds four tatsu parser singletons (module globals / class
   attrs). tatsu parsers keep mutable rule/AST stacks across ``parse()``;
   concurrent reset/step across sessions corrupts them nondeterministically
   (``IndexError: pop from empty list``, spurious ``FailedToken``).
3. ``fast_downward.translate`` (pure-python PDDL->SAS, runs during env load
   and reset) is a CLI port full of module globals — concurrent loads crash
   with ``KeyError`` deep in operator translation. The C side is safe: each
   env loads its own copy of libdownward.so. Loads/resets therefore serialize
   on TW_LOAD_LOCK; the cost is near-zero because translate is GIL-bound
   pure python anyway.
4. tatsu's PEG recursion overflows default worker-thread stacks (512 KB on
   macOS); the server sets ``threading.stack_size`` before spawning its
   executor threads.
"""

import threading

# Guards every textworld env load/reset (fast_downward.translate, see above).
TW_LOAD_LOCK = threading.Lock()

_PATCHED = False


class _PerThread:
    """Proxy that materializes one delegate per thread (parsers, converters)."""

    def __init__(self, factory):
        self._factory = factory
        self._tls = threading.local()

    def _obj(self):
        obj = getattr(self._tls, "obj", None)
        if obj is None:
            obj = self._factory()
            self._tls.obj = obj
        return obj

    def parse(self, *args, **kwargs):
        return self._obj().parse(*args, **kwargs)

    def walk(self, *args, **kwargs):
        return self._obj().walk(*args, **kwargs)


def apply_textworld_patches() -> None:
    """Idempotent; import-time cheap, so call before every env construction."""
    global _PATCHED
    if _PATCHED:
        return

    import sys

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

    import textworld.envs.pddl.logic as tw_pddl_logic
    import textworld.envs.pddl.textgen as tw_pddl_textgen
    import textworld.logic as tw_logic
    import textworld.textgen as tw_textgen

    def derive(self, context=None):
        context = context or self.context
        namespace = {"self": self, "context": context, **context["variables"]}
        value = eval(self.expression, vars(tw_pddl_textgen), namespace)  # noqa: S307 - upstream grammar eval
        return [tw_pddl_textgen.TerminalSymbol(value)]

    tw_pddl_textgen.EvalSymbol.derive = derive

    tw_pddl_textgen._PARSER = _PerThread(
        lambda: tw_pddl_textgen.CSGParser(semantics=tw_pddl_textgen.CSGModelBuilderSemantics(), parseinfo=True)
    )
    tw_pddl_logic._PARSER = _PerThread(
        lambda: tw_pddl_logic.PddlLogicParser(semantics=tw_pddl_logic.PddlLogicModelBuilderSemantics(), parseinfo=True)
    )
    tw_logic._PARSER = _PerThread(
        lambda: tw_logic.GameLogicParser(semantics=tw_logic.GameLogicModelBuilderSemantics(), parseinfo=True)
    )
    tw_textgen.TextGrammar._PARSER = _PerThread(
        lambda: tw_textgen.TextGrammarParser(semantics=tw_textgen.TextGrammarModelBuilderSemantics(), parseinfo=True)
    )
    tw_textgen.TextGrammar._CONVERTER = _PerThread(lambda: tw_textgen._Converter())

    _PATCHED = True
