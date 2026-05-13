# spikeon-llm-router

A local LLM routing proxy that sits in front of Ollama and Gemini. Clients speak the OpenAI-compatible chat completions API; the router picks the right model based on prompt classification and streams the response back.

## What it does

- **Routing** — classifies each prompt (code, reasoning, writing, memory, finance, Google context) and selects the appropriate local or cloud model
- **Gemini integration** — pulls context from Google Drive (Bills spreadsheet) and Gmail before forwarding finance/email queries to Gemini
- **Agent path** — orchestrator/worker models bypass classification and get tool directory injection with think-token filtering
- **Task decomposition** — multi-part prompts are split, run in sequence, then synthesized
- **Conversation logging** — every request/response is posted to [lancedb-agent-memory](https://github.com/spikeon/lancedb-agent-memory) for later retrieval

## Quick start

```sh
# Build
go build -o llm-router ./cmd/llm-router/

# Run (default port 11435)
PORT=11435 ./llm-router
```

Requires a local [Ollama](https://ollama.com) instance at `http://localhost:11434`.

## Model map

| Key | Model | Use |
|---|---|---|
| snappy | qwen2.5:0.5b | Trivial / factual / short |
| fast | gemma4:e2b | General chat |
| memory-fast | gemma4:e2b | Memory/preference saving (tool-capable) |
| coder | mistral:7b | Code / debug / math |
| balanced | gemma4:e4b | Write / summarize / explain |
| thinker | qwen3.6:latest | Deep reasoning / analysis |
| smart | gemma4:31b | Hard / long / complex |
| orchestrator | qwen3.6:latest | Agent orchestration (think-tokens filtered) |
| worker | qwen3.6:latest | Heavy tool-calling subagent (think-tokens filtered) |
| gemini | gemini-2.0-flash | Google cloud (Drive / Sheets / Gmail) |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |
| `GET` | `/v1/models` | List models |
| `GET` | `/v1/models/{id}` | Get model |
| `GET` | `/api/tags` | Ollama-style tag list |
| `GET` | `/v1/props` | Router metadata |
| `GET` | `/version` | Version string |

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `11435` | Listen port |
| `GEMINI_API_KEY` | — | Gemini API key (preferred) |
| `GEMINI_AUTH_JSON` | `~/.hermes/auth.json` | Path to JSON credential pool for Gemini key |
| `GOOGLE_TOKEN_PATH` | `~/.config/spikeon-router/google_token.json` | OAuth2 token file for Drive/Gmail |
| `GOOGLE_CREDENTIALS_PATH` | `~/.config/spikeon-router/google_credentials.json` | OAuth2 client credentials file |
| `CONV_INGEST_URL` | `http://127.0.0.1:3847/conv/log` | Conversation log ingest endpoint |

### Google OAuth setup

1. Create an OAuth2 client in [Google Cloud Console](https://console.cloud.google.com) with Drive, Sheets, and Gmail readonly scopes.
2. Download the credentials JSON and place it at `GOOGLE_CREDENTIALS_PATH`.
3. Run the OAuth flow once to produce `GOOGLE_TOKEN_PATH`.

## Project layout

```
cmd/llm-router/       # binary entry point
internal/
  config/             # model map, keywords, constants
  router/             # prompt classification and decomposition
  ollama/             # Ollama HTTP client + SSE streaming + ThinkFilter
  gemini/             # Gemini client + Google Drive/Gmail context
  convlog/            # fire-and-forget conversation log client
  handlers/           # HTTP handlers (one file per endpoint)
deployments/          # systemd service unit
```

## Running as a systemd service

```sh
cp deployments/spikeon-llm-router.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now spikeon-llm-router.service
```
