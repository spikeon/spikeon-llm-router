"""
Fire-and-forget conversation logger.
POSTs to the lancedb-agent-memory MCP server's HTTP ingest endpoint (port 3847).
That server is the sole LanceDB owner; this module has no DB dependency.
"""
import json
import uuid
import datetime
import threading
import urllib.request
import urllib.error

INGEST_URL = "http://127.0.0.1:3847/conv/log"


def _post(row: dict) -> None:
    try:
        data = json.dumps(row).encode()
        req = urllib.request.Request(
            INGEST_URL, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception as e:
        print(f"[conv_log] WARNING: {e}")


class ConversationLogger:
    def log(self, *, prompt, response, model_key, model_name, latency_ms,
            frustrated=False, decomposed=False, task_index=-1, total_tasks=-1,
            finish_reason="stop", has_tool_calls=False) -> None:
        _post({
            "id":             str(uuid.uuid4()),
            "ts":             datetime.datetime.utcnow().isoformat() + "Z",
            "prompt":         str(prompt or ""),
            "response":       str(response or ""),
            "model_key":      str(model_key or ""),
            "model_name":     str(model_name or ""),
            "latency_ms":     float(latency_ms),
            "frustrated":     bool(frustrated),
            "decomposed":     bool(decomposed),
            "task_index":     int(task_index),
            "total_tasks":    int(total_tasks),
            "finish_reason":  str(finish_reason or "stop"),
            "has_tool_calls": bool(has_tool_calls),
        })

    def log_async(self, **kwargs) -> None:
        threading.Thread(target=self.log, kwargs=kwargs, daemon=True).start()


_logger: ConversationLogger | None = None


def get_logger() -> ConversationLogger:
    global _logger
    if _logger is None:
        _logger = ConversationLogger()
    return _logger
