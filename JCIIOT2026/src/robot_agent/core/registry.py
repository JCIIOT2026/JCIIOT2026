"""Registry for robot skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from robot_agent.core.types import ExecutionContext, SkillResult


class Skill(Protocol):
    name: str
    description: str

    def can_handle(self, task: str) -> bool:
        ...

    def run(self, context: ExecutionContext) -> SkillResult:
        ...


@dataclass
class SkillRegistry:
    _skills: dict[str, Skill] = field(default_factory=dict)

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, skill_name: str) -> Skill | None:
        return self._skills.get(skill_name)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def find(self, task: str) -> Skill | None:
        for skill in self._skills.values():
            if skill.can_handle(task):
                return skill
        return None
