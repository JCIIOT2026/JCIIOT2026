"""Shared types for the robot agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SkillResult:
    skill_name: str
    success: bool
    message: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentStep:
    task: str
    skill_name: str
    result: SkillResult


@dataclass(slots=True)
class ExecutionContext:
    task: str
    metadata: dict[str, Any] = field(default_factory=dict)
