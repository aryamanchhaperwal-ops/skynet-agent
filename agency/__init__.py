"""SKYNET agency layer — the modular core that turns goals into recorded runs.

Public surface (see ``docs/SKYNET_CORE.md``):

- :class:`agency.loop.SkynetCore` — the controlled agent cycle
  (GOAL → OBSERVE → PLAN → ACT → RECORD → EVALUATE → COMPLETE).
- :func:`agency.bootstrap.build_core` — composition root wiring default
  implementations (deterministic planner/evaluator, God's Eye perception,
  database or memory storage).
- Extension points: :class:`agency.planner.Planner`,
  :class:`agency.actions.base.Action` / ``ActionRegistry``,
  :class:`agency.evaluator.Evaluator`,
  :class:`agency.perception.PerceptionAdapter`,
  :class:`agency.experience.ExperienceSink`,
  :class:`agency.trace.TraceSink`.

This package deliberately contains no LLM calls, no web access and no
self-modification: those are later phases behind the same interfaces.
"""

__version__ = "0.1.0"
