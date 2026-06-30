"""Task-domain knowledge base — all content loaded from markdown files.

No hardcoded knowledge lives here.  Everything is read dynamically from
``knowledge/`` (locked) and ``team_submission/knowledge/`` (team) at
planning time via :class:`KnowledgeManager`.
"""

from __future__ import annotations


def as_prompt_context() -> str:
    """Return empty — all knowledge is injected via KnowledgeManager."""
    return ""
