"""Hermes memory provider: spikeon-agent-memory (conversation log + LanceDB semantic memory).

Reads the same SQLite conversation log and LanceDB HTTP API as
https://github.com/spikeon/spikeon-agent-memory — no MCP subprocess.

Install: symlink this directory to ``$HERMES_HOME/plugins/spikeon_agent_memory``,
then ``hermes config set memory.provider spikeon_agent_memory`` and restart Hermes.

Config: ``$HERMES_HOME/spikeon_agent_memory.json`` (optional) plus env vars
(``LANCEDB_URL``, ``SPIKEON_CONV_DB_DIR``, ``EMBEDDING_*``, …) matching spikeon-agent-memory.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from . import backend

logger = logging.getLogger(__name__)


CONV_RECENT_SCHEMA = {
    "name": "spikeon_conv_recent",
    "description": (
        "List recent LLM router / agent conversations logged to spikeon-agent-memory "
        "(prompt + response), newest first. Use when the user refers to earlier chats, "
        "what we said before, or past sessions with the router."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max rows (default 15)"},
            "model_key": {"type": "string", "description": "Filter by router model key (e.g. snappy, orchestrator)"},
            "hours_ago": {"type": "number", "description": "Only rows newer than N hours ago (0 = all)"},
        },
    },
}

CONV_SEARCH_SCHEMA = {
    "name": "spikeon_conv_search",
    "description": (
        "Full-text style search across logged prompts and responses. "
        "Use for 'when did we discuss X', 'what did I ask about Y'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Substring to find in prompt or response"},
            "limit": {"type": "integer", "description": "Max rows (default 15)"},
            "model_key": {"type": "string", "description": "Optional filter by model_key"},
        },
        "required": ["text"],
    },
}

CONV_STATS_SCHEMA = {
    "name": "spikeon_conv_stats",
    "description": "Summary stats for the router conversation log: totals, per-model counts, avg latency.",
    "parameters": {"type": "object", "properties": {}},
}

MEMORY_RECALL_SCHEMA = {
    "name": "spikeon_memory_recall",
    "description": (
        "Semantic search in the spikeon LanceDB memory table (same as MCP memory_recall). "
        "Use for durable facts/preferences stored via spikeon_memory_save or MCP, not raw chat logs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language query"},
            "limit": {"type": "integer", "description": "Max hits (default 8)"},
        },
        "required": ["query"],
    },
}

MEMORY_SAVE_SCHEMA = {
    "name": "spikeon_memory_save",
    "description": (
        "Upsert a semantic memory row (embed + LanceDB). Mirrors MCP memory_save."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short label"},
            "text": {"type": "string", "description": "Body to retrieve later"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags",
            },
            "source": {"type": "string", "description": "Provenance (path, ticket id, etc.)"},
            "id": {"type": "string", "description": "Optional stable id (UUID). Omit to auto-generate."},
        },
        "required": ["title", "text"],
    },
}

MEMORY_DELETE_SCHEMA = {
    "name": "spikeon_memory_delete",
    "description": "Delete a semantic memory row from LanceDB by id.",
    "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Row id from recall/save"}},
        "required": ["id"],
    },
}


class SpikeonAgentMemoryProvider(MemoryProvider):
    """Bridge Hermes to local spikeon-agent-memory storage."""

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}
        self._hermes_home = ""
        self._prefetch_semantic = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None
        self._identity = "hermes"

    @property
    def name(self) -> str:
        return "spikeon_agent_memory"

    def is_available(self) -> bool:
        try:
            from hermes_constants import get_hermes_home

            h = str(get_hermes_home())
            cfg = backend.load_merged_config(h)
            dbp = backend.conv_db_path(cfg)
            if dbp.is_file():
                return True
            # Allow semantic-only setups
            return bool(str(cfg.get("lancedb_url", "")).strip())
        except Exception:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from hermes_constants import display_hermes_home

        default_dir = backend._home_default_conv_dir()
        return [
            {
                "key": "conv_db_dir",
                "description": "Directory containing memory.db (SQLite conversation log)",
                "default": default_dir,
            },
            {
                "key": "lancedb_url",
                "description": "lancedb-connector HTTP base URL",
                "default": "http://127.0.0.1:3030",
            },
            {
                "key": "memory_table",
                "description": "LanceDB table name (TABLE_NAME in spikeon-agent-memory)",
                "default": "cursor_agent_memory",
            },
            {
                "key": "mirror_turns_to_conv_log",
                "description": "POST each Hermes user/assistant turn to CONV_INGEST_URL (may duplicate llm-router logs)",
                "default": "false",
                "choices": ["true", "false"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = __import__("pathlib").Path(hermes_home) / "spikeon_agent_memory.json"
        cur = {}
        if path.is_file():
            try:
                cur = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
        cur.update({k: v for k, v in values.items() if v is not None})
        path.write_text(json.dumps(cur, indent=2), encoding="utf-8")

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home

        self._hermes_home = str(get_hermes_home())
        self._cfg = backend.load_merged_config(self._hermes_home)
        self._identity = str(kwargs.get("agent_identity") or "hermes")
        m = self._cfg.get("mirror_turns_to_conv_log")
        if isinstance(m, str):
            self._cfg["mirror_turns_to_conv_log"] = m.lower() in ("1", "true", "yes")
        plim = self._cfg.get("prefetch_semantic_limit", 0)
        try:
            self._cfg["prefetch_semantic_limit"] = int(plim)
        except (TypeError, ValueError):
            self._cfg["prefetch_semantic_limit"] = 0

    def system_prompt_block(self) -> str:
        dbp = backend.conv_db_path(self._cfg)
        ldb = str(self._cfg.get("lancedb_url", ""))
        bits = [
            "# Spikeon agent memory",
            "Connected to the same storage as spikeon-agent-memory (local SQLite conversation log + LanceDB via HTTP).",
        ]
        if dbp.is_file():
            bits.append(f"Conversation log: `{dbp}`")
        else:
            bits.append(f"Conversation log DB not found at `{dbp}` — conv_* tools will error until logging runs.")
        bits.append(f"LanceDB connector: `{ldb}` table `{self._cfg.get('memory_table')}`")
        bits.append(
            "When the user refers to **prior router/agent chats**, call `spikeon_conv_search` or "
            "`spikeon_conv_recent` first. For **saved semantic memories** (facts/preferences), use "
            "`spikeon_memory_recall` / `spikeon_memory_save`."
        )
        return "\n".join(bits)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        with self._prefetch_lock:
            result = self._prefetch_semantic
            self._prefetch_semantic = ""
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        lim = int(self._cfg.get("prefetch_semantic_limit") or 0)
        if lim <= 0 or not (query or "").strip():
            return

        cfg = self._cfg

        def _run() -> None:
            try:
                vec = backend.embed_text(cfg, query)
                raw = backend.lancedb_search(cfg, vec, min(lim, 10))
                if not raw:
                    return
                lines = []
                for row in backend.format_search_results(raw):
                    t = row.get("title") or ""
                    body = (row.get("text") or "")[:240]
                    sc = row.get("score", 0)
                    lines.append(f"- ({sc:.2f}) **{t}** — {body}")
                with self._prefetch_lock:
                    self._prefetch_semantic = (
                        "## Spikeon semantic memory (prefetch)\n" + "\n".join(lines)
                    )
            except Exception as e:
                logger.debug("spikeon prefetch: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="spikeon-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._cfg.get("mirror_turns_to_conv_log"):
            return
        url = str(self._cfg.get("conv_ingest_url") or "")
        if not url:
            return

        payload = {
            "prompt": user_content[:100_000],
            "response": assistant_content[:100_000],
            "model_key": "hermes",
            "model_name": self._identity,
            "latency_ms": 0.0,
            "frustrated": False,
            "decomposed": False,
            "task_index": -1,
            "total_tasks": -1,
            "finish_reason": "stop",
            "has_tool_calls": False,
        }

        def _sync() -> None:
            try:
                backend.post_conv_ingest(url, payload)
            except Exception as e:
                logger.warning("spikeon conv mirror: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=3.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="spikeon-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            CONV_RECENT_SCHEMA,
            CONV_SEARCH_SCHEMA,
            CONV_STATS_SCHEMA,
            MEMORY_RECALL_SCHEMA,
            MEMORY_SAVE_SCHEMA,
            MEMORY_DELETE_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        cfg = self._cfg
        try:
            if tool_name == "spikeon_conv_recent":
                rows = backend.conv_recent(
                    cfg,
                    int(args.get("limit") or 15),
                    str(args.get("model_key") or ""),
                    float(args.get("hours_ago") or 0),
                )
                return json.dumps({"ok": True, "count": len(rows), "conversations": rows}, ensure_ascii=False)

            if tool_name == "spikeon_conv_search":
                rows = backend.conv_search(
                    cfg,
                    str(args.get("text") or ""),
                    int(args.get("limit") or 15),
                    str(args.get("model_key") or ""),
                )
                return json.dumps({"ok": True, "count": len(rows), "conversations": rows}, ensure_ascii=False)

            if tool_name == "spikeon_conv_stats":
                stats = backend.conv_stats(cfg)
                return json.dumps({"ok": True, **stats}, ensure_ascii=False)

            if tool_name == "spikeon_memory_recall":
                q = str(args.get("query") or "").strip()
                if not q:
                    return tool_error("query is required")
                lim = int(args.get("limit") or 8)
                vec = backend.embed_text(cfg, q)
                raw = backend.lancedb_search(cfg, vec, max(1, min(lim, 50)))
                results = backend.format_search_results(raw)
                return json.dumps({"ok": True, "results": results}, ensure_ascii=False)

            if tool_name == "spikeon_memory_save":
                title = str(args.get("title") or "").strip()
                text = str(args.get("text") or "").strip()
                if not title or not text:
                    return tool_error("title and text are required")
                tags = args.get("tags") or []
                if not isinstance(tags, list):
                    tags = []
                tags = [str(t) for t in tags]
                source = str(args.get("source") or "")
                raw_id = args.get("id")
                pid = str(raw_id).strip() if raw_id else str(uuid.uuid4())
                input_text = f"{title}\n\n{text}"
                vec = backend.embed_text(cfg, input_text)
                row = backend.build_memory_row(cfg, pid, vec, title, text, tags, source)
                backend.lancedb_upsert(cfg, row)
                return json.dumps({"ok": True, "id": pid}, ensure_ascii=False)

            if tool_name == "spikeon_memory_delete":
                rid = str(args.get("id") or "").strip()
                if not rid:
                    return tool_error("id is required")
                backend.lancedb_delete(cfg, rid)
                return json.dumps({"ok": True, "deleted": rid}, ensure_ascii=False)

        except FileNotFoundError as e:
            return tool_error(str(e))
        except Exception as e:
            logger.exception("spikeon tool error")
            return tool_error(str(e))

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)


def register(ctx) -> None:
    ctx.register_memory_provider(SpikeonAgentMemoryProvider())
