"""Knowledge base manager — reads, deduplicates, classifies, and serves
documents from a ``knowledge/`` folder for the robot agent.

Documents are stored as plain-text files (``.txt``, ``.md``) in the
knowledge folder.  The manager:

* **Deduplicates** — SHA256 content hash; identical documents are skipped.
* **Classifies** — keyword-based auto-classification into categories
  (sop, reference, rule, manual, general).
* **Manages** — list, search, add, remove, reload, and export as
  prompt-ready context for the LLM planner.

Usage::

    from robot_agent.core.knowledge_manager import KnowledgeManager

    mgr = KnowledgeManager("knowledge/")
    mgr.reload()                     # scan folder
    ctx = mgr.as_prompt_context()    # compact summary for LLM
    results = mgr.search("搬运")     # fuzzy search
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── document model ────────────────────────────────────────────


@dataclass(slots=True)
class KnowledgeDoc:
    """One knowledge document."""

    doc_id: str          # SHA256 hex digest of normalized content
    title: str           # extracted or user-supplied title
    category: str        # auto-classified category
    tags: list[str]      # user / auto tags
    content: str         # raw text content
    source_file: str     # relative path inside knowledge folder
    added_at: str        # ISO timestamp
    size_bytes: int      # byte length of raw content


# ── classification rules ──────────────────────────────────────

# (category_key, label, [(weight, keyword), ...])
# weight 5 = document-type signal (title-level); weight 1 = content hint
CLASSIFICATION_RULES: list[tuple[str, str, list[tuple[int, str]]]] = [
    ("sop", "Standard Operating Procedure", [
        (10, "标准操作规程"),
        (10, "SOP"),
        (8, "操作规程"),
        (8, "作业流程"),
        (6, "标准操作"),
        (6, "任务编号"),
        (6, "步骤一"),
        (6, "步骤二"),
        (6, "步骤三"),
        (6, "步骤四"),
        (6, "步骤五"),
        (6, "步骤六"),
        (6, "步骤七"),
        (5, "标准搬运流程"),
        (5, "搬运作业"),
        (5, "拾取操作"),
        (5, "放置操作"),
        (4, "执行夹取"),
        (4, "执行放置"),
        (3, "确认夹取"),
        (3, "确认放置"),
        (3, "前置条件"),
    ]),
    ("reference", "Reference Document", [
        (8, "reference"),
        (8, "API"),
        (5, "接口"),
        (5, "配置参数"),
        (4, "config"),
        (3, "参考"),
        (3, "参数"),
        (3, "配置"),
    ]),
    ("rule", "Rules & Constraints", [
        (8, "约束规则"),
        (8, "安全规则"),
        (6, "禁止"),
        (6, "必须"),
        (5, "不得"),
        (5, "警告"),
        (4, "约束"),
        (4, "规则"),
        (3, "安全"),
        (3, "注意"),
        (3, "碰撞检测"),
    ]),
    ("manual", "Operation Manual", [
        (8, "操作手册"),
        (6, "部署"),
        (6, "安装说明"),
        (5, "使用指南"),
        (4, "手册"),
        (4, "说明"),
        (4, "指南"),
        (3, "使用"),
        (3, "安装"),
        (3, "启动"),
    ]),
    ("inventory", "Inventory / Scene", [
        (6, "库存"),
        (5, "物料清单"),
        (5, "工位布局"),
        (4, "场景"),
        (4, "地图"),
        (3, "物料"),
        (3, "工位"),
        (2, "进料口"),
        (2, "出料口"),
    ]),
]


def _classify(title: str, content: str) -> tuple[str, list[str]]:
    """Auto-classify *content* into a category and extract keyword tags.

    Uses weighted keyword matching.  Title-level signals (e.g. "SOP",
    "标准操作规程") carry higher weight than generic content hints
    (e.g. "物料", "安全").
    """
    text = f"{title} {content[:2000]}"
    text_lower = text.lower()

    best_cat = "general"
    best_score = 0
    auto_tags: list[str] = []

    for cat_key, _cat_label, weighted_keywords in CLASSIFICATION_RULES:
        score = 0
        for weight, kw in weighted_keywords:
            if kw.lower() in text_lower:
                score += weight
                if kw not in auto_tags:
                    auto_tags.append(kw)
        if score > best_score:
            best_score = score
            best_cat = cat_key

    return best_cat, auto_tags[:8]


def _extract_title(file_path: Path, content: str) -> str:
    """Extract a title from the first heading or first non-empty line."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Markdown / plain heading
        heading_match = re.match(r"^#{1,4}\s+(.+)$", stripped)
        if heading_match:
            return heading_match.group(1).strip()
        # First substantial line
        if len(stripped) > 3:
            return stripped[:120]
    return file_path.stem.replace("_", " ").strip()


def _content_hash(content: str) -> str:
    """SHA256 of whitespace-normalized content."""
    normalized = re.sub(r"\s+", " ", content.strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ── manager ───────────────────────────────────────────────────


class KnowledgeManager:
    """Manages a folder of knowledge documents for the robot agent.

    Parameters:
        root: Path to the knowledge folder.
        index_file: Name of the JSON index file inside *root*.
    """

    def __init__(self, root: str | Path = "knowledge", index_file: str = "_index.json") -> None:
        self._root = Path(root)
        self._index_path = self._root / index_file
        self._docs: dict[str, KnowledgeDoc] = {}   # doc_id → doc
        self._by_source: dict[str, str] = {}        # source_file → doc_id

        # Supported text file extensions
        self._text_exts = {".txt", ".md", ".rst", ".sop"}

    # ── public API ─────────────────────────────────────────

    @property
    def root(self) -> Path:
        return self._root

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def reload(self) -> int:
        """Scan the knowledge folder, dedup & index all documents.

        Returns the number of **new** documents added.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        added = 0

        # Load existing index so we don't lose manual metadata
        existing = self._load_index()

        for file_path in sorted(self._root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self._text_exts:
                continue
            if file_path.name.startswith("_") or file_path.name.startswith("."):
                continue

            rel = str(file_path.relative_to(self._root).as_posix())

            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("Skipping non-UTF-8 file: %s", rel)
                continue
            if not content.strip():
                continue

            doc_id = _content_hash(content)

            # Dedup: skip if already known by hash
            if doc_id in self._docs:
                logger.debug("Dedup skip (hash match): %s", rel)
                continue

            # Dedup: skip if source file already tracked
            prev_doc_id = self._by_source.get(rel)
            if prev_doc_id and prev_doc_id in self._docs:
                logger.debug("Dedup skip (source match): %s", rel)
                continue

            # Restore manual metadata from index, or auto-classify
            meta = existing.get(rel, {})
            title = meta.get("title") or _extract_title(file_path, content)
            category = meta.get("category") or "general"
            tags = list(meta.get("tags") or [])
            if not category or category == "general":
                category, auto_tags = _classify(title, content)
                tags = list(dict.fromkeys(tags + auto_tags))  # dedup preserve order

            doc = KnowledgeDoc(
                doc_id=doc_id,
                title=title,
                category=category,
                tags=tags,
                content=content,
                source_file=rel,
                added_at=meta.get("added_at") or _now_iso(),
                size_bytes=len(content.encode("utf-8")),
            )
            self._docs[doc_id] = doc
            self._by_source[rel] = doc_id
            added += 1

        if added:
            logger.info("KnowledgeManager: %d new document(s) loaded", added)
            self._save_index()
        return added

    def list_docs(self, *, category: str | None = None, tag: str | None = None) -> list[dict[str, Any]]:
        """Return a summary list of all (or filtered) documents."""
        result: list[dict[str, Any]] = []
        for doc in self._docs.values():
            if category and doc.category != category:
                continue
            if tag and tag not in doc.tags:
                continue
            result.append(self._doc_summary(doc))
        result.sort(key=lambda d: d["title"])
        return result

    def search(self, query: str, *, top_n: int = 10) -> list[dict[str, Any]]:
        """Fuzzy search documents by *query* (title + content substring match)."""
        q = query.lower()
        scored: list[tuple[int, KnowledgeDoc]] = []
        for doc in self._docs.values():
            score = 0
            # title match (weighted higher)
            title_lower = doc.title.lower()
            if q in title_lower:
                score += 10
            # content match
            content_lower = doc.content.lower()
            score += content_lower.count(q)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._doc_summary(doc) for _, doc in scored[:top_n]]

    def get_doc(self, doc_id: str) -> KnowledgeDoc | None:
        """Get a single document by ID."""
        return self._docs.get(doc_id)

    def get_doc_by_source(self, source_file: str) -> KnowledgeDoc | None:
        """Get a document by its relative source path."""
        doc_id = self._by_source.get(source_file)
        if doc_id:
            return self._docs.get(doc_id)
        return None

    def add_doc(self, title: str, content: str, *, category: str = "", tags: list[str] | None = None) -> str:
        """Add a document programmatically. Returns its doc_id.

        If the content matches an existing document (dedup), returns the
        existing doc_id without creating a duplicate.
        """
        doc_id = _content_hash(content)
        if doc_id in self._docs:
            logger.info("Dedup: document already exists as '%s'", self._docs[doc_id].title)
            return doc_id

        if not category:
            category, auto_tags = _classify(title, content)
            tags = list(dict.fromkeys((tags or []) + auto_tags))
        else:
            tags = tags or []

        # Write to a source file
        safe_name = re.sub(r"[^\w\-]+", "_", title.strip().lower())[:60]
        source_file = f"{safe_name}.md"
        file_path = self._root / source_file
        if file_path.exists():
            source_file = f"{safe_name}_{doc_id[:8]}.md"
            file_path = self._root / source_file

        file_path.write_text(content, encoding="utf-8")

        doc = KnowledgeDoc(
            doc_id=doc_id,
            title=title,
            category=category,
            tags=tags,
            content=content,
            source_file=source_file,
            added_at=_now_iso(),
            size_bytes=len(content.encode("utf-8")),
        )
        self._docs[doc_id] = doc
        self._by_source[source_file] = doc_id
        self._save_index()
        logger.info("Document added: '%s' (%s) → %s", title, category, source_file)
        return doc_id

    def remove_doc(self, doc_id: str) -> bool:
        """Remove a document by ID. Returns True if removed."""
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return False
        self._by_source.pop(doc.source_file, None)
        # Delete source file
        file_path = self._root / doc.source_file
        if file_path.exists():
            file_path.unlink()
        self._save_index()
        logger.info("Document removed: '%s' (%s)", doc.title, doc.source_file)
        return True

    def update_doc_meta(self, doc_id: str, *, title: str | None = None, category: str | None = None, tags: list[str] | None = None) -> bool:
        """Update metadata for an existing document."""
        doc = self._docs.get(doc_id)
        if doc is None:
            return False
        if title is not None:
            doc.title = title
        if category is not None:
            doc.category = category
        if tags is not None:
            doc.tags = tags
        self._save_index()
        return True

    def as_prompt_context(
        self,
        *,
        category: str | None = None,
        max_chars: int = 4000,
        exclude_sources: set[str] | None = None,
    ) -> str:
        """Return a compact knowledge summary for LLM prompt injection."""
        lines: list[str] = ["## 知识库文档"]

        docs = sorted(self._docs.values(), key=lambda d: (d.category, d.title))
        if category:
            docs = [d for d in docs if d.category == category]
        if exclude_sources:
            docs = [d for d in docs if d.source_file not in exclude_sources]

        if not docs:
            return ""

        # Group by category
        from collections import defaultdict
        by_cat: dict[str, list[KnowledgeDoc]] = defaultdict(list)
        for doc in docs:
            by_cat[doc.category].append(doc)

        total_chars = 0
        for cat_key, cat_docs in sorted(by_cat.items()):
            cat_label = _cat_label(cat_key)
            lines.append(f"\n### {cat_label}")
            for doc in cat_docs:
                # Truncate content to keep total under max_chars
                snippet = doc.content[:600].replace("\n", " ").strip()
                if len(doc.content) > 600:
                    snippet += "…"
                entry = f"- **{doc.title}** [{', '.join(doc.tags[:5])}] — {snippet}"
                total_chars += len(entry)
                if total_chars > max_chars:
                    lines.append(f"- … ({len(docs) - len(lines)} more documents)")
                    return "\n".join(lines)
                lines.append(entry)

        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        """Return statistics about the knowledge base."""
        by_cat: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        total_bytes = 0
        for doc in self._docs.values():
            by_cat[doc.category] = by_cat.get(doc.category, 0) + 1
            for tag in doc.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1
            total_bytes += doc.size_bytes
        return {
            "total_docs": len(self._docs),
            "total_size_kb": round(total_bytes / 1024, 1),
            "by_category": dict(sorted(by_cat.items())),
            "top_tags": dict(sorted(by_tag.items(), key=lambda x: x[1], reverse=True)[:15]),
            "root": str(self._root),
        }

    # ── internal ───────────────────────────────────────────

    def _doc_summary(self, doc: KnowledgeDoc) -> dict[str, Any]:
        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "category": doc.category,
            "tags": doc.tags,
            "source_file": doc.source_file,
            "added_at": doc.added_at,
            "size_bytes": doc.size_bytes,
            "preview": doc.content[:200].replace("\n", " ").strip(),
        }

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load persisted metadata from _index.json."""
        if not self._index_path.exists():
            return {}
        try:
            import json
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            return data.get("documents", {}) if isinstance(data, dict) else {}
        except Exception:
            logger.warning("Failed to load index, starting fresh")
            return {}

    def _save_index(self) -> None:
        """Persist metadata to _index.json."""
        import json
        index: dict[str, dict[str, Any]] = {}
        for doc in self._docs.values():
            index[doc.source_file] = {
                "title": doc.title,
                "category": doc.category,
                "tags": doc.tags,
                "added_at": doc.added_at,
            }
        payload = {
            "updated_at": _now_iso(),
            "document_count": len(index),
            "documents": index,
        }
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _cat_label(key: str) -> str:
    for cat_key, label, _ in CLASSIFICATION_RULES:
        if cat_key == key:
            return label
    return "Other"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
