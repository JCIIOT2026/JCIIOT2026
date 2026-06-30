"""Standard output schema utilities for planner and agent.

This module defines a stable JSON structure that the planner should produce
and helper functions to normalise LLM outputs into that structure.

Also provides ``repair_json()`` — a best-effort fixer for common LLM JSON
mistakes (markdown fences, trailing commas, truncation, single quotes, etc.).
"""

from __future__ import annotations

import json
import re
import logging
from datetime import datetime, timezone
from typing import Any

from robot_agent.core.registry import SkillRegistry

logger = logging.getLogger(__name__)

STANDARD_VERSION = "2.0"


# ── helpers ───────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ── JSON repair ───────────────────────────────────────────

# precompiled patterns for repair
_MD_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SINGLE_QUOTE_KEY_RE = re.compile(r"'([^']+)'\s*:")
_SINGLE_QUOTE_STR_RE = re.compile(r":\s*'([^']*)'")


def repair_json(raw: str) -> dict[str, Any] | None:
    """Attempt to salvage a malformed JSON string from an LLM.

    Handles the most common failure modes:
    1. Markdown code fences (``````json ... ```` ``` ``)
    2. Trailing commas before ``}`` or ``]``
    3. Single-quoted keys / string values
    4. Text before / after the JSON object
    5. Truncation — missing closing brace(s)

    Returns the parsed dict, or *None* if every strategy fails.
    """
    if not raw or not raw.strip():
        return None

    candidates: list[str] = []

    # Strategy 0: raw as-is
    candidates.append(raw.strip())

    # Strategy 1: extract from markdown fences
    m = _MD_FENCE_RE.search(raw)
    if m:
        candidates.append(m.group(1).strip())

    # Strategy 2: find the outermost {...} span
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for cand in candidates:
        # 2a) strip BOM / zero-width chars
        cand = cand.lstrip("﻿")

        # 2b) fix trailing commas
        cand = _TRAILING_COMMA_RE.sub(r"\1", cand)

        # 2c) try direct parse
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass

        # 2d) replace single quotes (conservative — only keys and simple values)
        try:
            fixed = _SINGLE_QUOTE_KEY_RE.sub(r'"\1":', cand)
            fixed = _SINGLE_QUOTE_STR_RE.sub(r': "\1"', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 2e) truncation repair — add missing closing braces/brackets
        try:
            return _repair_truncation(cand)
        except json.JSONDecodeError:
            continue

    return None


def _repair_truncation(s: str) -> dict[str, Any]:
    """Try to close unclosed braces/brackets in truncated JSON.

    Tries common close-patterns because the correct order depends on
    nesting (which simple counts can't recover).  For the small brace
    counts typical in planner output (≤10) this is fast and reliable.
    """
    braces = s.count("{") - s.count("}")
    brackets = s.count("[") - s.count("]")

    # close unclosed string
    in_string = False
    escaped = False
    for ch in s:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        s += '"'

    # generate plausible close-orders
    suffixes: list[str] = []

    # 1) brackets-first:  ]]]... }}}...
    suffixes.append("]" * brackets + "}" * braces)

    # 2) braces-first:    }}}... ]]]...
    suffixes.append("}" * braces + "]" * brackets)

    # 3) interleaved:  }] }] }] ...  (each bracket paired with a brace inside)
    paired = min(braces, brackets)
    suffixes.append("}]" * paired + "}" * (braces - paired) + "]" * (brackets - paired))

    # 4) interleaved reversed:  ]} ]} ]} ...
    suffixes.append("]}" * paired + "}" * (braces - paired) + "]" * (brackets - paired))

    for suffix in suffixes:
        try:
            return json.loads(s + suffix)
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("truncation repair failed", s, len(s))


# ── validation ────────────────────────────────────────────

def validate_plan_steps(
    plan: list[dict[str, Any]], registry: SkillRegistry
) -> list[str]:
    """Validate each step's ``skill_name`` against the registry.

    Returns a list of warning messages (empty = all good).  Does **not**
    raise — a missing skill is noted as a warning because the agent can
    still fall back via ``registry.find()``.
    """
    warnings: list[str] = []
    known = {s.name for s in registry.all()}
    for i, step in enumerate(plan):
        name = step.get("skill_name", "")
        if not name:
            warnings.append(f"Step {i}: empty skill_name")
        elif name not in known:
            warnings.append(
                f"Step {i}: skill_name '{name}' not in registry "
                f"(known: {sorted(known)})"
            )
    return warnings


# ── normalisation ─────────────────────────────────────────

def normalize_planner_output(
    raw: str | dict[str, Any], task: str, registry: SkillRegistry
) -> dict[str, Any]:
    """Normalize raw planner output (string or dict) into the standard schema.

    Result keys:
      - version
      - primary_skill
      - understanding
      - reason
      - plan: list of {skill_name, description, inputs,
              preconditions, expected_output, timeout, retries}
      - explanation
      - timestamp
      - raw
      - warnings (list[str])
    """
    details: dict[str, Any]
    if isinstance(raw, str):
        try:
            details = json.loads(raw)
        except json.JSONDecodeError:
            repaired = repair_json(raw)
            if repaired is not None:
                details = repaired
            else:
                details = {"raw": raw}
    else:
        details = dict(raw)

    # ── build output ──────────────────────────────────────
    out: dict[str, Any] = {}
    out["version"] = details.get("version", STANDARD_VERSION)
    out["timestamp"] = details.get("timestamp", _now_iso())
    out["understanding"] = (
        details.get("understanding")
        or details.get("task_understanding")
        or str(task)
    )
    out["reason"] = (
        details.get("reason") or details.get("selection_reason") or ""
    )
    out["explanation"] = (
        details.get("explanation") or details.get("result_explanation") or ""
    )

    # ── normalise plan ────────────────────────────────────
    plan = details.get("plan")
    normalized_plan: list[dict[str, Any]] = []
    if isinstance(plan, list):
        for step in plan:
            if not isinstance(step, dict):
                continue
            skill_name = step.get("skill_name") or step.get("skill") or ""
            description = step.get("description") or step.get("task") or ""
            inputs = step.get("inputs") or {}
            preconditions = step.get("preconditions") or []
            expected = step.get("expected_output") or step.get("expected") or ""
            timeout = step.get("timeout") or step.get("duration") or None
            retries = step.get("retries") or 0
            # ensure types
            if not isinstance(preconditions, list):
                preconditions = [str(preconditions)]
            if not isinstance(inputs, dict):
                inputs = {}
            try:
                retries = int(retries)
            except (TypeError, ValueError):
                retries = 0
            try:
                timeout = float(timeout) if timeout is not None else None
            except (TypeError, ValueError):
                timeout = None

            normalized_plan.append(
                {
                    "skill_name": skill_name,
                    "description": description,
                    "inputs": inputs,
                    "preconditions": preconditions,
                    "expected_output": expected,
                    "timeout": timeout,
                    "retries": retries,
                }
            )

    # ── fallback: heuristic keyword matching ──────────────
    if not normalized_plan:
        lowered = task.lower()
        preferred = [
            "move",
            "pick_up",
            "place_down",
        ]
        for name in preferred:
            s = registry.get(name)
            if s and s.can_handle(lowered):
                normalized_plan.append(
                    {"skill_name": s.name, "description": task}
                )

        if not normalized_plan:
            for s in registry.all():
                if s.can_handle(lowered):
                    normalized_plan.append(
                        {"skill_name": s.name, "description": task}
                    )

    # final fallback
    if not normalized_plan:
        fallback = registry.find(task)
        if fallback:
            normalized_plan.append(
                {"skill_name": fallback.name, "description": task}
            )

    out["plan"] = normalized_plan
    out["primary_skill"] = (
        normalized_plan[0]["skill_name"]
        if normalized_plan
        else (details.get("skill_name") or "fallback")
    )

    # ── validate ───────────────────────────────────────────
    out["warnings"] = validate_plan_steps(normalized_plan, registry)

    # ── preserve raw for audit ─────────────────────────────
    out["raw"] = (
        raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    )
    return out
