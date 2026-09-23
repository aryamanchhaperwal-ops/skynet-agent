"""SKYNET experimentation + self-evaluation engine (Phase P6).

Public surface:

- :class:`~agency.experiments.models.Experiment` — one controlled comparison
- :class:`~agency.experiments.models.Hypothesis` — testable claim with frozen criteria
- :class:`~agency.experiments.registry.StrategyRegistry` — versioned strategy artifacts
- :class:`~agency.experiments.runner.ExperimentRunner` — the only executor
- :class:`~agency.experiments.experiments_store.ExperimentRegistry` — history/persistence
- :class:`~agency.experiments.evaluation.evaluate_experiment` — deterministic verdicts

Safety posture (blueprint §7): experiments operate on *registered
strategies and data* through a sandboxed :class:`TrialContext`; they can
never execute, reference, or modify source code, and the LLM advisor can
never decide a verdict.
"""

from agency.experiments.evaluation import LLMExperimentAdvisor, evaluate_experiment
from agency.experiments.experiments_store import ExperimentRegistry
from agency.experiments.memory_bridge import ORIGIN_EXPERIMENT, record_experiment
from agency.experiments.metrics import aggregate, compare
from agency.experiments.models import (
    ArmSummary,
    ComparisonReport,
    Experiment,
    ExperimentEvaluation,
    ExperimentStatus,
    ExperimentSummary,
    Hypothesis,
    MetricDefinition,
    SuccessCriterion,
    TargetRef,
    TrialResult,
    Verdict,
)
from agency.experiments.registry import (
    STRATEGY_CAPABILITIES,
    Procedure,
    Strategy,
    StrategyRegistry,
)
from agency.experiments.runner import ExperimentRunner
from agency.experiments.sandbox import Sandbox, SandboxError, TrialContext

__all__ = [
    "EXPERIMENT_MEMORY_ORIGIN",
    "ORIGIN_EXPERIMENT",
    "STRATEGY_CAPABILITIES",
    "ArmSummary",
    "ComparisonReport",
    "Experiment",
    "ExperimentEvaluation",
    "ExperimentRegistry",
    "ExperimentRunner",
    "ExperimentStatus",
    "ExperimentSummary",
    "Hypothesis",
    "LLMExperimentAdvisor",
    "MetricDefinition",
    "Procedure",
    "Sandbox",
    "SandboxError",
    "Strategy",
    "StrategyRegistry",
    "SuccessCriterion",
    "TargetRef",
    "TrialContext",
    "TrialResult",
    "Verdict",
    "aggregate",
    "compare",
    "evaluate_experiment",
    "record_experiment",
]
