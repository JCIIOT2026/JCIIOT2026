"""
Competition Skill Interface Contract
=====================================
This file defines the EXACT interface that competition teams must implement.
DO NOT modify this file — it is part of the platform core.

Teams implement :class:`BaseSkill` subclasses in ``team_submission/skills/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# =========================================================================
#  Data contracts — identical to robot_agent.core.types
# =========================================================================

@dataclass(slots=True)
class SkillResult:
    """Returned by every ``skill.run()`` call.

    Attributes:
        skill_name: Name of the skill that produced this result.
        success: Whether the skill completed its objective.
        message: Human-readable summary (shown in UI).
        payload: Arbitrary structured data (target, waypoints, errors…).
    """
    skill_name: str
    success: bool
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionContext:
    """Passed into ``skill.run()``.  Carries the task and planner metadata.

    Attributes:
        task: The natural-language command for this step.
        metadata: Planner-supplied information (inputs, step_index, attempt).
    """
    task: str
    metadata: dict[str, Any] = field(default_factory=dict)


# =========================================================================
#  Skill base class — teams subclass this
# =========================================================================

class BaseSkill:
    """Abstract skill that every competition skill must extend.

    Subclasses register themselves with a unique *name*, a human-readable
    *description*, and a set of *keywords* used for LLM fuzzy matching.

    Usage::

        class MyPickSkill(BaseSkill):
            def __init__(self, *, backend):
                super().__init__(
                    name="pick_up",
                    description="Grasp an object at the named station",
                    keywords=("pick", "grasp", "抓"),
                )
                self._backend = backend

            def run(self, ctx: ExecutionContext) -> SkillResult:
                target = ctx.metadata["inputs"]["target"]
                ok = self._backend.grasp_object_physics(target)
                return SkillResult(skill_name=self.name, success=ok,
                                   message=f"Grasp {'OK' if ok else 'FAIL'}: {target}")
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        keywords: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.description = description
        self.keywords = keywords

    def can_handle(self, task: str) -> bool:
        """Return True if *task* matches any of this skill's keywords."""
        task_lower = task.lower()
        return any(kw.lower() in task_lower for kw in self.keywords)

    def run(self, context: ExecutionContext) -> SkillResult:
        """Execute the skill.  Subclasses MUST override this."""
        raise NotImplementedError("Subclasses must implement run()")


# =========================================================================
#  Backend API — what skills can call on the simulation
# =========================================================================

class EnvBackend(Protocol):
    """Minimal protocol that the competition backend exposes to skills.

    All methods are thread-safe and may be called from any skill.
    """

    def get_base_pose(self) -> tuple:
        """Return ``(xy: np.ndarray, yaw: float)`` — robot world pose."""
        ...

    def follow_path(self, path) -> bool:
        """Navigate the mobile base along *path* (list of (x,y) waypoints).

        Returns True if the full path was reached.
        """
        ...

    def grasp_object_physics(self, source: str) -> bool:
        """Run the BC grasp policy at *source* station.

        Returns True if the object was successfully grasped and lifted.
        """
        ...

    def place_object_physics(self, target: str) -> bool:
        """Turn to face *target* station and place the held object.

        Returns True if placement succeeded.
        """
        ...

    @property
    def env(self):
        """Direct access to the robosuite MuJoCo environment (escape hatch)."""
        ...

    def capture_frame(self, camera: str = "birdview",
                      width: int = 640, height: int = 480):
        """Render one RGB frame from *camera*."""
        ...

    def get_available_crates(self) -> dict:
        """Return ``{port_name: obj_name}`` for objects still at input stations."""
        ...

    def start_recording(self) -> None:
        """Begin accumulating trajectory frames."""
        ...

    def stop_recording(self) -> list:
        """Stop recording and return accumulated frames."""
        ...

    def save_trajectory(self, path) -> str:
        """Save accumulated trajectory as JSON."""
        ...


# =========================================================================
#  Checklist for competition teams
# =========================================================================
"""
1. Create ``team_submission/skills/my_pick_up.py`` extending ``BaseSkill``.
2. Implement ``run(context) -> SkillResult``.
3. Call ``self._backend.grasp_object_physics(target)`` or use ``self._backend.env`` directly.
4. Put your trained checkpoint in ``team_submission/models/``.
5. Configure ``team_submission/config.yaml`` with your model path.
6. Submit ``team_submission/`` as a zip file.
"""
