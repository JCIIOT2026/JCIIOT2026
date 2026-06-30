"""Skill library — wired to a real or simulated backend.

All skills require a backend; there is no mock / no-op fallback.
"""

from __future__ import annotations

import numpy as np

from robot_agent.core.memory import InMemoryStore
from robot_agent.core.scene_context import SceneContext
from robot_agent.environments.base import EnvBackend
from robot_agent.skills.base import BaseSkill
from robot_agent.skills.move import MoveSkill
from robot_agent.skills.pick_up import PickUpSkill
from robot_agent.skills.place_down import PlaceDownSkill
from robot_agent.skills.record_trajectory import RecordTrajectorySkill
from robot_agent.skills.analyze_supply import AnalyzeSupplySkill
from robot_agent.skills.knowledge_mgr import KnowledgeMgrSkill
from robot_agent.skills.memory_mgr import MemoryMgrSkill
from robot_agent.skills.read_document import ReadDocumentSkill


def wired_skills(
    backend: EnvBackend,
    scene_context: SceneContext,
    grid: np.ndarray,
    *,
    path_spacing: float = 0.35,
    memory_store: InMemoryStore | None = None,
) -> list[BaseSkill]:
    """Return skills wired to a real (or simulated) backend."""
    skills: list[BaseSkill] = [
        MoveSkill(
            backend=backend,
            scene_context=scene_context,
            grid=grid,
            path_spacing=path_spacing,
        ),
        PickUpSkill(backend=backend, scene_context=scene_context),
        PlaceDownSkill(backend=backend, scene_context=scene_context),
        AnalyzeSupplySkill(
            backend=backend,
            scene_context=scene_context,
            grid=grid,
            path_spacing=path_spacing,
        ),
        RecordTrajectorySkill(backend=backend),
        KnowledgeMgrSkill(knowledge_root="knowledge"),
        ReadDocumentSkill(),
    ]
    if memory_store is not None:
        skills.append(MemoryMgrSkill(store=memory_store))
    return skills
