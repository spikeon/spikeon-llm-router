#!/usr/bin/env python3
"""
Overnight orchestrator tester.
Cycles through candidate models, swaps each into the orchestrator slot,
restarts the router, fires the standard Discord test prompt via Hermes CLI,
and records whether the model attempted any Discord MCP tool calls.

Results: /tmp/model_test_results.md
"""

import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

CONFIG_PATH  = Path("/home/spikeon/Dev/spikeon-llm-router/config.py")
RESULTS_PATH = Path("/tmp/model_test_results.md")
HERMES_CLI   = [
    "/home/spikeon/.hermes/hermes-agent/venv/bin/python",
    "-m", "hermes_cli.main",
]
HERMES_ENV = {**os.environ, "HERMES_HOME": "/home/spikeon/.hermes"}

TEST_PROMPT = (
    "How far back in this Discord channel can you access? 1304105121768149113  "
    "If you don't know how, let's find you a skill or a plugin, just let me know that.  "
    "I just installed the Discord MCP for you, so PLEASE try."
)

# (ollama_model_name, short_label)
MODELS = [
    ("qwen3-coder:latest",    "qwen3-coder"),
    ("deepseek-coder-v2:16b", "deepseek-coder-v2"),
    ("devstral:latest",       "devstral"),
    ("gemma4:31b",            "gemma4-31b"),
    ("gemma4:e4b",            "gemma4-e4b"),
    ("mistral:7b",            "mistral-7b"),
    ("qwen3.6:latest",        "qwen3.6"),
]

DOWNLOAD_TIMEOUT = 4 * 3600  # 4 hours per model before skipping
TEST_TIMEOUT     = 180        # seconds per Hermes chat call


# ── helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def is_downloaded(model_name: str) -> bool:
    r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    base = model_name.split(":")[0]
    return base in r.stdout


def wait_for_model(model_name: str) -> bool:
    if is_downloaded(model_name):
        return True
    log(f"  waiting for {model_name} to finish downloading ...")
    deadline = time.time() + DOWNLOAD_TIMEOUT
    while time.time() < deadline:
        time.sleep(60)
        if is_downloaded(model_name):
            log(f"  {model_name} ready")
            return True
    return False


def swap_orchestrator(ollama_name: str):
    text = CONFIG_PATH.read_text()
    new_text = re.sub(
        r'("orchestrator":\s*\{[^}]*?"name":\s*")[^"]+(")',
        rf'\g<1>{ollama_name}\g<2>',
        text,
        flags=re.DOTALL,
    )
    CONFIG_PATH.write_text(new_text)


def restart_router():
    subprocess.run(
        ["systemctl", "--user", "restart", "spikeon-llm-router.service"],
        check=True,
    )
    time.sleep(12)  # wait for uvicorn startup


def recent_logs(n: int = 120) -> str:
    r = subprocess.run(
        ["journalctl", "--user", "-u", "spikeon-llm-router.service",
         "-n", str(n), "--no-pager", "--output=short"],
        capture_output=True, text=True,
    )
    return r.stdout


def run_test() -> dict:
    before = recent_logs(10)  # baseline snapshot

    try:
        result = subprocess.run(
            HERMES_CLI + ["chat", "-q", TEST_PROMPT, "--max-turns", "6"],
            capture_output=True, text=True,
            timeout=TEST_TIMEOUT, env=HERMES_ENV,
        )
        output = (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        output = "(hermes chat timed out)"

    logs = recent_logs(120)

    # Collect actual tool call names from router log lines (format: "    tool_call: name(args)")
    tool_calls = re.findall(r"tool_call:\s*(\S+?)\(", logs)

    # Did the model call a real Discord MCP tool?
    tried_discord = any("discord" in t.lower() for t in tool_calls)

    # Finish reasons
    finish_reasons = re.findall(r"finish(?:_reason)?=(\w+)", logs)

    return {
        "tried_discord": tried_discord,
        "tool_calls":    list(dict.fromkeys(tool_calls)),  # dedup, preserve order
        "finish_reasons": list(dict.fromkeys(finish_reasons)),
        "response":      output[:500] if output else "(empty)",
    }


def write_result(label: str, ollama_name: str, result: dict):
    status = "✅ TRIED DISCORD" if result["tried_discord"] else "❌ did not try Discord"
    block = (
        f"\n## {label} (`{ollama_name}`)\n"
        f"**Result:** {status}  \n"
        f"**Tools called:** {', '.join(result['tool_calls']) or 'none'}  \n"
        f"**Finish reasons:** {', '.join(result['finish_reasons']) or 'unknown'}  \n"
        f"**Response snippet:**\n```\n{result['response']}\n```\n"
        f"---\n"
    )
    with RESULTS_PATH.open("a") as f:
        f.write(block)
    log(f"  → {status}  tools={result['tool_calls']}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_PATH.write_text(
        f"# Orchestrator Discord Tool Test\n"
        f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"**Prompt:** `{TEST_PROMPT[:100]}...`\n\n"
        f"---\n"
    )
    log("Starting overnight model sweep")

    for ollama_name, label in MODELS:
        log(f"\n=== {label} ({ollama_name}) ===")

        if not wait_for_model(ollama_name):
            log(f"SKIP: {ollama_name} not downloaded after {DOWNLOAD_TIMEOUT//3600}h")
            with RESULTS_PATH.open("a") as f:
                f.write(f"\n## {label} — SKIPPED (download timeout)\n---\n")
            continue

        log(f"  swapping orchestrator → {ollama_name}")
        swap_orchestrator(ollama_name)
        restart_router()

        log("  running test prompt ...")
        result = run_test()
        write_result(label, ollama_name, result)

    log("\n=== ALL DONE ===")
    log(f"Results: {RESULTS_PATH}")
    print(f"\n{'='*60}")
    print(RESULTS_PATH.read_text())


if __name__ == "__main__":
    main()
