---
name: spikeon-agent-memory-recall
description: >-
  When the user refers to past conversations with the LLM router, Hermes, or Cursor
  agents logged via spikeon-agent-memory — e.g. "what did we discuss", "last time
  you said", "when did I ask about X" — query the conversation log (and optionally
  semantic memory) before answering. Use the spikeon-agent-memory MCP tools or
  Hermes spikeon_conv_* / spikeon_memory_* tools.
---

# spikeon-agent-memory recall

## When this applies

- User references **earlier chats**, **prior sessions**, **what we talked about**, **router / Hermes history**, or **logged prompts**.
- You need **ground truth** from storage, not model memory.

## Data sources (same backend)

| Layer | Storage | Access |
|-------|---------|--------|
| Conversation log | SQLite `memory.db` → table `router_conversation_log` | MCP `conv_recent`, `conv_search`, `conv_stats` — or Hermes tools `spikeon_conv_*` when that memory provider is active |
| Semantic memory | LanceDB via lancedb-connector | MCP `memory_recall`, `memory_save`, … — or Hermes `spikeon_memory_recall` / `spikeon_memory_save` |

Default SQLite directory on many installs: `~/.exoworks/lancedb-agent-memory/memory.db`.

## Workflow

1. **Interpret intent** — narrow time range or topic (topic → `conv_search` with `text`; recency → `conv_recent` with `limit` / `hours_ago`; model filter → `model_key` if user named a router key like `snappy` / `orchestrator`).
2. **Query first** — prefer `conv_search` for keywords; use `conv_recent` for “last N exchanges”.
3. **Summarize** — cite timestamps, model keys, and short excerpts; do not invent turns that are not in the tool results.
4. **Semantic facts** — if the user asks about **saved preferences / long-term facts** (not raw chat), use `memory_recall` (MCP) or `spikeon_memory_recall` (Hermes) with a natural-language query.

## Hermes

If the user runs [Hermes Agent](https://hermes-agent.nousresearch.com/) with the **`spikeon_agent_memory`** memory provider, the same operations are available as builtin tools (`spikeon_conv_search`, etc.) without MCP. Install: `contrib/hermes-plugins/spikeon_agent_memory` → symlink to `$HERMES_HOME/plugins/spikeon_agent_memory`; `hermes config set memory.provider spikeon_agent_memory`.

## If tools are missing

Say that spikeon-agent-memory MCP must be enabled (or Hermes memory provider installed), and the ingest service / router logging must be running so rows exist.
