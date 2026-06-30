"""Simple workflow for executing a task end-to-end."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot_agent.core.agent import RobotAgent
from robot_agent.core.scene_context import SceneContext
from robot_agent.core.types import SkillResult
from robot_agent.environments.base import EnvBackend


@dataclass
class TaskFlow:
    """Thin wrapper that builds and runs a RobotAgent.

    Usage::

        flow = TaskFlow(backend=backend, scene_context=scene_ctx, grid=grid)
        result = flow.execute("把1号进料口的物体送到3号出料口")
    """

    backend: EnvBackend
    scene_context: SceneContext
    grid: np.ndarray
    path_spacing: float = 0.35

    agent: RobotAgent = field(init=False)

    def __post_init__(self) -> None:
        self.agent = RobotAgent(
            backend=self.backend,
            scene_context=self.scene_context,
            grid=self.grid,
            path_spacing=self.path_spacing,
        )

    def execute(self, task: str) -> SkillResult:
        return self.agent.run(task)
