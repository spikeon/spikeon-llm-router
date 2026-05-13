# spikeon_agent_memory — Hermes memory provider

Bridges [Hermes Agent](https://github.com/NousResearch/hermes-agent) to [spikeon-agent-memory](https://github.com/spikeon/spikeon-agent-memory) storage:

- **Conversation log** — same SQLite file as the MCP server (`memory.db` / `router_conversation_log`).
- **Semantic memory** — same LanceDB table via **lancedb-connector** HTTP (`/v1/tables/...`), with embeddings through your existing **Ollama** or OpenAI-compatible endpoint.

Hermes tools exposed:

| Tool | Purpose |
|------|---------|
| `spikeon_conv_recent` | Recent logged prompts/responses (newest first) |
| `spikeon_conv_search` | Substring search on prompt + response |
| `spikeon_conv_stats` | Totals / per-model / avg latency |
| `spikeon_memory_recall` | Vector search (MCP `memory_recall` equivalent) |
| `spikeon_memory_save` | Upsert memory row (MCP `memory_save` equivalent) |
| `spikeon_memory_delete` | Delete by row id |

## Install

Hermes loads user memory plugins from **`$HERMES_HOME/plugins/<name>/`** (see `plugins/memory/__init__.py`).

```bash
mkdir -p "$HERMES_HOME/plugins"
ln -sfn /path/to/spikeon-llm-router/contrib/hermes-plugins/spikeon_agent_memory \
  "$HERMES_HOME/plugins/spikeon_agent_memory"
```

Use your real `HERMES_HOME` (often `~/.hermes` or `~/.hermes/profiles/<profile>`).

## Activate

```bash
hermes config set memory.provider spikeon_agent_memory
hermes memory status   # optional
```

Restart the Hermes CLI or gateway so the provider reloads.

## Configuration

Non-secrets can live in **`$HERMES_HOME/spikeon_agent_memory.json`**. Environment variables override the file (same names as spikeon-agent-memory where possible):

| Key / env | Default | Meaning |
|-----------|---------|---------|
| `conv_db_dir` / `SPIKEON_CONV_DB_DIR` | `~/.exoworks/lancedb-agent-memory` | Directory containing `memory.db` |
| `lancedb_url` / `LANCEDB_URL` | `http://127.0.0.1:3030` | lancedb-connector base URL |
| `memory_table` / `SPIKEON_MEMORY_TABLE`, `TABLE_NAME` | `cursor_agent_memory` | Vector table name |
| `embedding_url` / `EMBEDDING_URL` | `http://127.0.0.1:11434/api/embeddings` | Embeddings endpoint |
| `embedding_model` / `EMBEDDING_MODEL` | `all-minilm:l6-v2` | Model id |
| `embedding_backend` / `EMBEDDING_BACKEND` | `ollama` | `ollama` or `openai` |
| `embedding_dim` / `EMBEDDING_DIM` | `384` | Vector dimension |
| `mirror_turns_to_conv_log` / `SPIKEON_MIRROR_TURNS` | `false` | POST each Hermes turn to `conv_ingest_url` (can duplicate llm-router logs) |
| `conv_ingest_url` / `CONV_INGEST_URL` | `http://127.0.0.1:3847/conv/log` | Used only if mirroring |
| `prefetch_semantic_limit` / `SPIKEON_PREFETCH_SEMANTIC` | `0` | If &gt; 0, semantic prefetch before each turn |

Run `hermes memory setup` and pick this provider for a guided first run (if listed).

## Requirements

- **spikeon-agent-memory** services you already use: daemon ingest (3847), lancedb-connector (3030), embedding API (e.g. Ollama).
- Python stdlib only inside the plugin (`urllib` + `sqlite3`); Hermes already ships the rest of the stack.

## Availability

`is_available()` is true if `memory.db` exists under `conv_db_dir`, or if `lancedb_url` is non-empty (semantic-only setups).
