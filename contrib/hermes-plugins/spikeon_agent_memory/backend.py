"""HTTP + SQLite helpers for spikeon-agent-memory parity (LanceDB connector + conv log).

Uses stdlib ``urllib`` only so the plugin imports under any Hermes/runtime Python.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


def _home_default_conv_dir() -> str:
    return str(Path.home() / ".exoworks" / "lancedb-agent-memory")


def load_merged_config(hermes_home: str) -> dict[str, Any]:
    """Env vars override values from $HERMES_HOME/spikeon_agent_memory.json."""
    from hermes_constants import get_hermes_home

    path = Path(hermes_home or str(get_hermes_home())) / "spikeon_agent_memory.json"
    data: dict[str, Any] = {
        "conv_db_dir": os.environ.get("SPIKEON_CONV_DB_DIR", _home_default_conv_dir()),
        "lancedb_url": os.environ.get("LANCEDB_URL", "http://127.0.0.1:3030"),
        "memory_table": os.environ.get(
            "SPIKEON_MEMORY_TABLE",
            os.environ.get("TABLE_NAME", os.environ.get("MEMORY_TABLE", "cursor_agent_memory")),
        ),
        "embedding_url": os.environ.get("EMBEDDING_URL", "http://127.0.0.1:11434/api/embeddings"),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "all-minilm:l6-v2"),
        "embedding_backend": os.environ.get("EMBEDDING_BACKEND", "ollama"),
        "embedding_api_key": os.environ.get("EMBEDDING_API_KEY", ""),
        "embedding_dim": int(os.environ.get("EMBEDDING_DIM", "384")),
        "mirror_turns_to_conv_log": os.environ.get("SPIKEON_MIRROR_TURNS", "").lower()
        in ("1", "true", "yes"),
        "conv_ingest_url": os.environ.get("CONV_INGEST_URL", "http://127.0.0.1:3847/conv/log"),
        "prefetch_semantic_limit": int(os.environ.get("SPIKEON_PREFETCH_SEMANTIC", "0")),
    }
    if path.is_file():
        try:
            file_cfg = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(file_cfg, dict):
                for k, v in file_cfg.items():
                    if v is not None and v != "":
                        data[k] = v
        except Exception as e:
            logger.debug("spikeon_agent_memory.json: %s", e)
    if os.environ.get("LANCEDB_URL"):
        data["lancedb_url"] = os.environ["LANCEDB_URL"]
    if os.environ.get("SPIKEON_CONV_DB_DIR"):
        data["conv_db_dir"] = os.environ["SPIKEON_CONV_DB_DIR"]
    return data


def conv_db_path(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg["conv_db_dir"])) / "memory.db"


def escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str],
    timeout: float,
) -> tuple[int, Any]:
    data_bytes = None
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = resp.getcode() or 200
            if not raw:
                return code, None
            return code, json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            parsed = json.loads(raw) if raw else None
        except Exception:
            parsed = raw
        raise RuntimeError(f"HTTP {e.code}: {raw[:400] if isinstance(raw, str) else parsed}") from e


def embed_text(cfg: dict[str, Any], text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        raise ValueError("cannot embed empty text")
    backend = str(cfg.get("embedding_backend", "ollama")).lower()
    url = str(cfg["embedding_url"])
    model = str(cfg["embedding_model"])
    headers = {"Content-Type": "application/json"}
    key = cfg.get("embedding_api_key") or ""
    if key:
        headers["Authorization"] = f"Bearer {key}"

    if backend == "openai":
        _, body = _http_json(
            "POST",
            url,
            {"model": model, "input": text},
            headers,
            120.0,
        )
        data = (body or {}).get("data") or []
        if not data or not data[0].get("embedding"):
            raise RuntimeError("embedding API returned empty vector")
        return [float(x) for x in data[0]["embedding"]]

    _, body = _http_json(
        "POST",
        url,
        {"model": model, "prompt": text},
        headers,
        120.0,
    )
    emb = (body or {}).get("embedding")
    if not emb:
        raise RuntimeError("embedding API returned empty vector")
    return [float(x) for x in emb]


def _lancedb_base(cfg: dict[str, Any]) -> str:
    return str(cfg.get("lancedb_url", "http://127.0.0.1:3030")).rstrip("/")


def _table_path_segment(cfg: dict[str, Any]) -> str:
    return quote(str(cfg["memory_table"]), safe="")


def lancedb_search(cfg: dict[str, Any], vector: list[float], limit: int) -> list[dict[str, Any]]:
    t = _table_path_segment(cfg)
    url = f"{_lancedb_base(cfg)}/v1/tables/{t}/search"
    try:
        _, body = _http_json("POST", url, {"vector": vector, "limit": limit}, {"Content-Type": "application/json"}, 60.0)
    except RuntimeError as e:
        if str(e).startswith("HTTP 404"):
            return []
        raise
    return list((body or {}).get("results") or [])


def lancedb_upsert(cfg: dict[str, Any], row: dict[str, Any]) -> None:
    t = _table_path_segment(cfg)
    url = f"{_lancedb_base(cfg)}/v1/tables/{t}/upsert"
    _http_json("POST", url, {"on": "id", "rows": [row]}, {"Content-Type": "application/json"}, 60.0)


def lancedb_delete(cfg: dict[str, Any], row_id: str) -> None:
    t = _table_path_segment(cfg)
    safe = str(row_id).replace("'", "''")
    url = f"{_lancedb_base(cfg)}/v1/tables/{t}/rows"
    data_bytes = json.dumps({"where": f"id = '{safe}'"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            if resp.getcode() and resp.getcode() >= 400:
                raise RuntimeError(f"HTTP {resp.getcode()}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code}: {raw[:400]}") from e


def conv_recent(cfg: dict[str, Any], limit: int, model_key: str, hours_ago: float) -> list[dict[str, Any]]:
    dbp = conv_db_path(cfg)
    if not dbp.is_file():
        raise FileNotFoundError(f"conversation database not found: {dbp}")
    if limit <= 0:
        limit = 20
    conds: list[str] = []
    args: list[Any] = []
    if model_key:
        conds.append("model_key = ?")
        args.append(model_key)
    if hours_ago > 0:
        cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=hours_ago * 3600)
        cutoff = cutoff_dt.isoformat().replace("+00:00", "Z")
        conds.append("ts >= ?")
        args.append(cutoff)
    q = (
        "SELECT id, ts, prompt, response, model_key, model_name, latency_ms, "
        "frustrated, decomposed, task_index, total_tasks, finish_reason, has_tool_calls "
        "FROM router_conversation_log"
    )
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    conn = sqlite3.connect(str(dbp))
    try:
        cur = conn.execute(q, args)
        rows = []
        for tup in cur.fetchall():
            rows.append(
                {
                    "id": tup[0],
                    "ts": tup[1],
                    "prompt": tup[2],
                    "response": tup[3],
                    "model_key": tup[4],
                    "model_name": tup[5],
                    "latency_ms": tup[6],
                    "frustrated": bool(tup[7]),
                    "decomposed": bool(tup[8]),
                    "task_index": tup[9],
                    "total_tasks": tup[10],
                    "finish_reason": tup[11],
                    "has_tool_calls": bool(tup[12]),
                }
            )
        return rows
    finally:
        conn.close()


def conv_search(cfg: dict[str, Any], text: str, limit: int, model_key: str) -> list[dict[str, Any]]:
    dbp = conv_db_path(cfg)
    if not dbp.is_file():
        raise FileNotFoundError(f"conversation database not found: {dbp}")
    if not (text or "").strip():
        raise ValueError("text is required")
    if limit <= 0:
        limit = 20
    esc = escape_like(text.strip())
    like = f"%{esc}%"
    conds = ["(prompt LIKE ? ESCAPE '\\' OR response LIKE ? ESCAPE '\\')"]
    args: list[Any] = [like, like]
    if model_key:
        conds.append("model_key = ?")
        args.append(model_key)
    args.append(limit)
    q = (
        "SELECT id, ts, prompt, response, model_key, model_name, latency_ms, "
        "frustrated, decomposed, task_index, total_tasks, finish_reason, has_tool_calls "
        "FROM router_conversation_log WHERE "
        + " AND ".join(conds)
        + " ORDER BY ts DESC LIMIT ?"
    )
    conn = sqlite3.connect(str(dbp))
    try:
        cur = conn.execute(q, args)
        rows = []
        for tup in cur.fetchall():
            rows.append(
                {
                    "id": tup[0],
                    "ts": tup[1],
                    "prompt": tup[2],
                    "response": tup[3],
                    "model_key": tup[4],
                    "model_name": tup[5],
                    "latency_ms": tup[6],
                    "frustrated": bool(tup[7]),
                    "decomposed": bool(tup[8]),
                    "task_index": tup[9],
                    "total_tasks": tup[10],
                    "finish_reason": tup[11],
                    "has_tool_calls": bool(tup[12]),
                }
            )
        return rows
    finally:
        conn.close()


def conv_stats(cfg: dict[str, Any]) -> dict[str, Any]:
    dbp = conv_db_path(cfg)
    out: dict[str, Any] = {"total": 0, "byModel": {}, "avgLatencyMs": 0, "newest": ""}
    if not dbp.is_file():
        return out
    conn = sqlite3.connect(str(dbp))
    try:
        cur = conn.execute(
            "SELECT model_key, latency_ms, ts FROM router_conversation_log",
        )
        by_model: dict[str, int] = {}
        total = 0
        total_lat = 0.0
        newest = ""
        for model_key, lat, ts in cur.fetchall():
            by_model[model_key] = by_model.get(model_key, 0) + 1
            total += 1
            total_lat += float(lat or 0)
            if ts and ts > newest:
                newest = ts
        out["total"] = total
        out["byModel"] = by_model
        out["avgLatencyMs"] = int(round(total_lat / total)) if total else 0
        out["newest"] = newest
        return out
    finally:
        conn.close()


def score_from_distance(dist: float) -> float:
    s = 1.0 - float(dist) / 2.0
    return max(0.0, min(1.0, s))


def format_search_results(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for r in raw:
        dist = r.get("_distance")
        score = score_from_distance(float(dist)) if dist is not None else 0.0
        tags_json = r.get("tags_json") or "[]"
        try:
            tags = json.loads(tags_json) if isinstance(tags_json, str) else tags_json
        except Exception:
            tags = []
        formatted.append(
            {
                "id": r.get("id"),
                "score": score,
                "title": r.get("title"),
                "text": r.get("text"),
                "tags": tags,
                "source": r.get("source"),
                "saved_at": r.get("saved_at"),
            },
        )
    return formatted


def build_memory_row(
    cfg: dict[str, Any],
    pid: str,
    vector: list[float],
    title: str,
    text: str,
    tags: list[str],
    source: str,
) -> dict[str, Any]:
    saved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "id": pid,
        "vector": vector,
        "title": title.strip(),
        "text": text.strip(),
        "tags_json": json.dumps(tags or []),
        "source": (source or "").strip(),
        "saved_at": saved_at,
        "extra_json": "{}",
    }


def post_conv_ingest(url: str, payload: dict[str, Any]) -> None:
    try:
        code, _ = _http_json("POST", url, payload, {"Content-Type": "application/json"}, 5.0)
        if code >= 400:
            logger.warning("conv ingest %s: status %s", url, code)
    except Exception as e:
        logger.warning("conv ingest %s: %s", url, e)
