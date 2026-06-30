"""Standard output types for the robot agent.

Everything the agent returns MUST use the types defined here.
When adding a field to agent output:
  1. Add it HERE first
  2. Update the call sites in agent.py
  3. Update the renderer in app.py

Design: plain dataclasses (no slots — must be subclass-friendly).
Every type has an ``as_dict()`` for serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── step output ───────────────────────────────────────────

@dataclass
class StepOutput:
    """The result of executing a single plan step."""

    skill: str
    description: str
    success: bool
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    expected_output: str = ""
    timeout: float | None = None
    retries: int = 0
    attempts: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "description": self.description,
            "success": self.success,
            "message": self.message,
            "payload": self.payload,
            "inputs": self.inputs,
            "preconditions": self.preconditions,
            "expected_output": self.expected_output,
            "timeout": self.timeout,
            "retries": self.retries,
            "attempts": self.attempts,
        }


# ── thinking metadata ─────────────────────────────────────

@dataclass
class ThinkingOutput:
    """Planner reasoning metadata attached to every result."""

    version: str = "2.0"
    task_understanding: str = ""
    selection_reason: str = ""
    execution_steps: list[str] = field(default_factory=list)
    plan_summary: list[dict[str, str]] = field(default_factory=list)
    result_explanation: str = ""

    @classmethod
    def from_details(
        cls,
        details: dict[str, Any],
        plan: list[dict[str, Any]] | None = None,
    ) -> ThinkingOutput:
        plan = plan or []
        return cls(
            version=details.get("version", "2.0"),
            task_understanding=details.get("understanding")
            or details.get("task_understanding") or "",
            selection_reason=details.get("reason") or "",
            execution_steps=[
                s.get("description") or s.get("task") or "" for s in plan
            ],
            plan_summary=[
                {"skill": s.get("skill_name", ""), "description": s.get("description", "")}
                for s in plan
            ],
            result_explanation=details.get("explanation") or "",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_understanding": self.task_understanding,
            "selection_reason": self.selection_reason,
            "execution_steps": self.execution_steps,
            "plan_summary": self.plan_summary,
            "result_explanation": self.result_explanation,
        }


# ── task output (the final result) ─────────────────────────

@dataclass
class TaskOutput:
    """The complete result of ``RobotAgent.run()``.

    This is the single canonical output type.  ``as_dict()`` produces
    the JSON-compatible dict consumed by the Streamlit UI.
    """

    skill_name: str = "composed"
    success: bool = True
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    steps: list[StepOutput] = field(default_factory=list)
    thinking: ThinkingOutput = field(default_factory=ThinkingOutput)
    planner_raw: str = ""
    plan_warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Full JSON-compatible dict for the UI."""
        out: dict[str, Any] = {
            "skill_name": self.skill_name,
            "success": self.success,
            "message": self.message,
            "steps": [s.as_dict() for s in self.steps],
            "thinking": self.thinking.as_dict(),
            "planner_raw": self.planner_raw,
            "plan_warnings": list(self.plan_warnings),
            "elapsed_ms": self.elapsed_ms,
        }
        # merge skill-level extras
        out.update(self.payload)
        return out

