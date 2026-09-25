"""SKYNET autonomous system integration (Phase P8).

Public surface:

- :class:`~agency.orchestrator.orchestrator.AutonomousOrchestrator` — the
  stage driver coordinating every finished phase
- :class:`~agency.orchestrator.models.AutonomousRun` — the persistent,
  resumable run record
- :class:`~agency.orchestrator.state_machine` — explicit validated stage
  transitions
- :mod:`agency.orchestrator.adapters` — the real-integration boundary
  (local deterministic adapters for tests/demos; P4/P5-backed adapters
  for real operation)
- :class:`~agency.orchestrator.store.RunStore` — cross-process-safe run
  persistence (pause/resume/cancel from another terminal)

Safety posture: external content is data (security module); budgets pause
rather than continue; state-changing improvements stay behind the P7
human approval gate; the orchestrator grants itself no new permissions.
"""

from agency.orchestrator.adapters import (
    CommsAdapter,
    LocalCommsAdapter,
    LocalResearchAdapter,
    ResearchAdapter,
    build_comms_adapter,
    build_research_adapter,
)
from agency.orchestrator.models import (
    ApprovalRequest,
    AutonomousRun,
    BudgetDecision,
    Evidence,
    IterationPlan,
    OrchestratorBudgets,
    RunStage,
    RunStatus,
    StageDecision,
)
from agency.orchestrator.orchestrator import AutonomousOrchestrator, OrchestratorError
from agency.orchestrator.security import (
    assert_no_control_payload,
    sanitize_provenance,
    scan_content,
)
from agency.orchestrator.state_machine import (
    InvalidTransitionError,
    allowed_stages,
    can_transition,
    require_transition,
)
from agency.orchestrator.store import RunStore

__all__ = [
    "ApprovalRequest",
    "AutonomousOrchestrator",
    "AutonomousRun",
    "BudgetDecision",
    "CommsAdapter",
    "Evidence",
    "InvalidTransitionError",
    "IterationPlan",
    "LocalCommsAdapter",
    "LocalResearchAdapter",
    "OrchestratorBudgets",
    "OrchestratorError",
    "ResearchAdapter",
    "RunStage",
    "RunStatus",
    "RunStore",
    "StageDecision",
    "allowed_stages",
    "assert_no_control_payload",
    "build_comms_adapter",
    "build_research_adapter",
    "can_transition",
    "require_transition",
    "sanitize_provenance",
    "scan_content",
]
