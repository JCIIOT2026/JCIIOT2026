"""In-memory history store with search, summary, and persistence.

The store records every ``AgentStep`` the agent executes and exposes
query methods used by the ``memory_mgr`` skill at runtime.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque

from robot_agent.core.types import AgentStep

logger = logging.getLogger(__name__)


@dataclass
class InMemoryStore:
    """Append-only history with search, recall, stats, and optional persistence.

    Parameters:
        limit: Max number of steps to retain (FIFO eviction).
        persist_path: If set, save/load history from this JSON file.
    """

    limit: int = 32
    persist_path: str | Path | None = None

    # ── post-init ──────────────────────────────────────────

    def __post_init__(self) -> None:
        self._history: Deque[AgentStep] = deque(maxlen=self.limit)
        self._step_counter: int = 0          # monotonic, survives eviction
        if self.persist_path:
            self._load()

    # ── core (used by agent) ───────────────────────────────

    def add(self, step: AgentStep) -> None:
        self._history.append(step)
        self._step_counter += 1
        if self.persist_path:
            self._save()

    def items(self) -> list[AgentStep]:
        return list(self._history)

    # ── search / recall ────────────────────────────────────

    def recall(self, query: str, *, top_n: int = 10) -> list[dict[str, Any]]:
        """Fuzzy search past steps by *query* (task, skill, message)."""
        q = query.lower()
        scored: list[tuple[int, AgentStep]] = []
        for step in self._history:
            score = 0
            if q in step.task.lower():
                score += 5
            if q in step.skill_name.lower():
                score += 3
            if q in step.result.message.lower():
                score += 2
            # payload deep search
            for v in step.result.payload.values():
                if isinstance(v, str) and q in v.lower():
                    score += 1
            if score > 0:
                scored.append((score, step))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._step_snapshot(s, score=sc) for sc, s in scored[:top_n]]

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the most recent *n* steps (newest first)."""
        items = list(self._history)
        items.reverse()
        return [self._step_snapshot(s) for s in items[:n]]

    def by_skill(self, skill_name: str) -> list[dict[str, Any]]:
        """Return all steps executed by a particular skill."""
        sn = skill_name.lower()
        return [
            self._step_snapshot(s)
            for s in self._history
            if sn in s.skill_name.lower()
        ]

    def failures(self) -> list[dict[str, Any]]:
        """Return only failed steps."""
        return [
            self._step_snapshot(s)
            for s in self._history
            if not s.result.success
        ]

    # ── summary ────────────────────────────────────────────

    def summarize(self) -> str:
        """Return a natural-language summary of recent activity."""
        if not self._history:
            return "Memory is empty. No operation records yet."

        items = list(self._history)
        total = len(items)
        success_count = sum(1 for s in items if s.result.success)
        fail_count = total - success_count

        # Last 5 actions
        recent_actions: list[str] = []
        for s in items[-5:]:
            status = "✓" if s.result.success else "✗"
            recent_actions.append(f"{status} {s.skill_name}: {s.result.message}")

        # Most used skills
        skill_counts: dict[str, int] = {}
        for s in items:
            skill_counts[s.skill_name] = skill_counts.get(s.skill_name, 0) + 1
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        lines = [
            f"Memory overview: {total} records (OK {success_count}, FAIL {fail_count})",
            f"Most used skills: {', '.join(f'{name}({cnt})' for name, cnt in top_skills)}",
            "Recent operations:",
        ]
        lines.extend(f"  {a}" for a in recent_actions)
        if fail_count > 0:
            lines.append(f"⚠ {fail_count} failed operations — call memory_mgr action=failures for details")
        return "\n".join(lines)

    # ── management ─────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return memory statistics."""
        items = list(self._history)
        skill_counts: dict[str, int] = {}
        total_payload_bytes = 0
        for s in items:
            skill_counts[s.skill_name] = skill_counts.get(s.skill_name, 0) + 1
            total_payload_bytes += len(json.dumps(s.result.payload, ensure_ascii=False, default=str).encode("utf-8"))
        return {
            "total_steps": len(items),
            "step_counter": self._step_counter,
            "limit": self.limit,
            "success_count": sum(1 for s in items if s.result.success),
            "fail_count": sum(1 for s in items if not s.result.success),
            "by_skill": dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)),
            "approx_payload_kb": round(total_payload_bytes / 1024, 1),
            "persist_path": str(self.persist_path) if self.persist_path else None,
        }

    def clear(self) -> int:
        """Clear all memory. Returns the number of steps removed."""
        count = len(self._history)
        self._history.clear()
        if self.persist_path:
            self._save()
        logger.info("Memory cleared: %d steps removed", count)
        return count

    def forget(self, query: str) -> int:
        """Remove steps whose task or skill_name matches *query*.  Returns count removed."""
        q = query.lower()
        keep: list[AgentStep] = []
        removed = 0
        for step in self._history:
            if q in step.task.lower() or q in step.skill_name.lower():
                removed += 1
            else:
                keep.append(step)
        self._history = deque(keep, maxlen=self.limit)
        if removed and self.persist_path:
            self._save()
        logger.info("Memory forget '%s': %d steps removed", query, removed)
        return removed

    # ── persistence ────────────────────────────────────────

    def _save(self) -> None:
        if not self.persist_path:
            return
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "step_counter": self._step_counter,
            "limit": self.limit,
            "steps": [
                {
                    "task": s.task,
                    "skill_name": s.skill_name,
                    "success": s.result.success,
                    "message": s.result.message,
                    "payload": s.result.payload,
                }
                for s in self._history
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def _load(self) -> None:
        if not self.persist_path:
            return
        path = Path(self.persist_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._step_counter = data.get("step_counter", 0)
            steps_data = data.get("steps", [])
            for s in steps_data[:self.limit]:
                from robot_agent.core.types import SkillResult
                step = AgentStep(
                    task=s.get("task", ""),
                    skill_name=s.get("skill_name", ""),
                    result=SkillResult(
                        skill_name=s.get("skill_name", ""),
                        success=s.get("success", False),
                        message=s.get("message", ""),
                        payload=s.get("payload", {}),
                    ),
                )
                self._history.append(step)
            logger.info("Memory loaded: %d steps from %s", len(self._history), self.persist_path)
        except Exception:
            logger.warning("Failed to load memory from %s, starting fresh", self.persist_path)

    # ── helpers ────────────────────────────────────────────

    @staticmethod
    def _step_snapshot(step: AgentStep, *, score: int | None = None) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "task": step.task,
            "skill_name": step.skill_name,
            "success": step.result.success,
            "message": step.result.message,
            "payload_keys": sorted(step.result.payload.keys()),
        }
        if score is not None:
            snap["score"] = score
        return snap
