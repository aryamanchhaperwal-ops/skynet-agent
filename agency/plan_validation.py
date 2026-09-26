"""Plan validation — a gate between a planner and the executor.

A plan is control input: whatever it names gets executed by
:meth:`agency.loop.SkynetCore._execute_step`. Deterministic planners are
code and their output is predictable; an LLM planner's output is a model
guess and must never be executed unvalidated. This module makes that
guarantee checkable:

INVALID PLAN
    → validation failure
    → deterministic safe fallback/replan (:class:`~agency.planner.DeterministicPlanner`
      restricted to the executable action set)
    → bounded retry (the fallback is validated once, no loop)
    → safe termination if it is still invalid.

The checks are intentionally mechanical: required fields, action
registration/allow-list membership, step budget, and a control-parameter
denylist (a step may not smuggle ``command``/``shell``/``eval``-shaped
parameters into an action). Nothing here reasons about content — content
is data, and data never becomes a step.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agency.actions.base import ActionSpec
from agency.planner import Plan

#: Parameter names that look like control/command smuggling. A model (or
#: external content that reached a model's prompt) must not be able to turn
#: an action's parameter dict into a command invocation.
_CONTROL_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "cmd",
        "command",
        "commands",
        "code",
        "eval",
        "exec",
        "execute",
        "shell",
        "subprocess",
        "script",
        "powershell",
        "bash",
    }
)


class PlanIssue(BaseModel):
    """One reason a plan was rejected."""

    code: str
    detail: str


class PlanValidation(BaseModel):
    """Structured verdict for one plan (never raises)."""

    valid: bool = True
    step_count: int = 0
    issues: list[PlanIssue] = Field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [f"{issue.code}: {issue.detail}" for issue in self.issues]


def validate_plan(
    plan: Plan,
    *,
    allowed_actions: frozenset[str] | None = None,
    registered_actions: frozenset[str] | None = None,
    max_steps: int | None = None,
) -> PlanValidation:
    """Validate a model-derived plan before it is executed.

    ``allowed_actions`` is the execution allow-list (the loop's
    ``action_allowlist`` gate); ``registered_actions`` is what the registry
    actually holds. Both are checked when supplied — a plan must name an
    action that exists *and* that the run is permitted to execute.
    """
    issues: list[PlanIssue] = []

    if not plan.steps:
        issues.append(
            PlanIssue(code="empty_plan", detail="plan contains no steps to execute")
        )
    if max_steps is not None and len(plan.steps) > max_steps:
        issues.append(
            PlanIssue(
                code="step_budget",
                detail=f"plan has {len(plan.steps)} steps; max allowed is {max_steps}",
            )
        )

    for index, step in enumerate(plan.steps):
        _validate_step(
            step,
            index=index,
            issues=issues,
            allowed_actions=allowed_actions,
            registered_actions=registered_actions,
        )

    return PlanValidation(valid=not issues, step_count=len(plan.steps), issues=issues)


def _validate_step(
    step: ActionSpec,
    *,
    index: int,
    issues: list[PlanIssue],
    allowed_actions: frozenset[str] | None,
    registered_actions: frozenset[str] | None,
) -> None:
    action_type = (step.type or "").strip()
    if not action_type:
        issues.append(
            PlanIssue(code="missing_action", detail=f"step {index} has an empty action type")
        )
        return
    if registered_actions is not None and action_type not in registered_actions:
        issues.append(
            PlanIssue(
                code="action_not_registered",
                detail=f"step {index} names unregistered action {action_type!r}",
            )
        )
    if allowed_actions and action_type not in allowed_actions:
        issues.append(
            PlanIssue(
                code="action_not_allowed",
                detail=f"step {index} names action {action_type!r} outside the allow-list",
            )
        )
    if not isinstance(step.params, dict):
        issues.append(
            PlanIssue(
                code="invalid_params",
                detail=f"step {index} ({action_type!r}) has non-object params",
            )
        )
        return
    control = sorted(
        {str(key).lower() for key in step.params} & _CONTROL_PARAM_KEYS
    )
    if control:
        issues.append(
            PlanIssue(
                code="control_param",
                detail=(
                    f"step {index} ({action_type!r}) carries control-shaped "
                    f"parameter(s) {control}; refusing to execute"
                ),
            )
        )


__all__ = ["PlanIssue", "PlanValidation", "validate_plan"]
