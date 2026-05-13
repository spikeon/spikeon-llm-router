# spikeon-llm-router

A local LLM routing proxy built in Go. Clients speak the [OpenAI chat completions API](https://platform.openai.com/docs/api-reference/chat); the router classifies each prompt and picks the right model — local via [Ollama](https://ollama.com) or cloud via [Gemini](https://ai.google.dev/) — then streams the response back.

## Features

### Prompt classification and routing

Every request is classified by keyword matching and heuristics before a model is selected. The priority order is:

1. **Finance / Google keywords** → Gemini (has access to your Drive/Sheets/Gmail)
2. **Reasoning keywords** (analyze, compare, tradeoffs…) → thinker model
3. **Personal info patterns** (memory keywords, "my X is", "I prefer") → memory-fast model
4. **Domain names / web references** → balanced model
5. **Code keywords** → coder model
6. **Long prompt** (>600 estimated tokens) → smart model
7. **Writing keywords** (summarize, rewrite, translate…) → balanced model
8. **Everything else** → fast model

### Caveman mode

All prompts are prefixed with a system instruction that compresses responses to maximum signal and zero fluff: no articles, no pleasantries, symbols over words, lists over paragraphs. The prompt is defined in [`internal/config/config.go`](internal/config/config.go) as `CavemanSystemPrompt`.

### Frustration detection

Before routing, each prompt is checked for frustration signals: swear words or a message where more than 50% of alphabetic characters are uppercase. Frustrated prompts are automatically upgraded to the `smart` model for a better answer.

### Task decomposition

Multi-part prompts are detected using a connector-word heuristic ("and then", "and also", "additionally", "furthermore"…) combined with sentence count. When triggered, the `snappy` model breaks the prompt into an ordered numbered list of atomic subtasks. Each subtask is routed and executed sequentially, with outputs from earlier steps available to later ones via `$variable` references. After all tasks complete, the `fast` model runs a **completion verification** step, checking each task response against the original request and outputting a `✓ / ✗` verdict.

### ThinkFilter

[Qwen3](https://ollama.com/library/qwen3) models emit chain-of-thought reasoning inside `<think>…</think>` tags before the actual response. The `ThinkFilter` in [`internal/ollama/think.go`](internal/ollama/think.go) is a stateful streaming filter that strips these tokens in real time so clients never see them. It handles tag boundaries that land mid-chunk.

### Google context injection

For prompts classified as finance or email-related, the router fetches live context from Google APIs before calling Gemini:

- **Google Drive + Sheets** — searches Drive for a spreadsheet named "Bills" and reads the full contents
- **Gmail** — searches for recent threads matching the prompt (up to 5, with subject/sender/snippet)

This context is injected into the Gemini system prompt, so answers about your bills or inbox are grounded in real data. Powered by [`golang.org/x/oauth2`](https://pkg.go.dev/golang.org/x/oauth2) and the [Google API Go client](https://github.com/googleapis/google-api-go-client).

### Agent path (orchestrator / worker)

Two model keys — `orchestrator` and `worker` — bypass the classifier entirely. They are designed for [Hermes](https://github.com/spikeon/hermes) agent loops where the calling agent already knows what model to use. These paths inject a tool directory into the system prompt and apply ThinkFilter so reasoning tokens are never exposed to the agent framework.

### Conversation logging

Every request and response (prompt, model used, latency, frustration flag, decomposition info, finish reason, tool calls) is posted asynchronously to [lancedb-agent-memory](https://github.com/spikeon/lancedb-agent-memory) via its HTTP ingest API. The logging is fire-and-forget: it never blocks or fails the main request. The ingest endpoint is configurable via `CONV_INGEST_URL`.

### SSE streaming proxy

Streaming responses from Ollama are proxied to the client using standard [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events). The Go handler uses `http.Flusher` to flush each chunk immediately, and `bufio.Scanner` to read SSE lines from Ollama line by line. ThinkFilter is applied inline for qwen3 models on the agent path.

---

## Quick start

```sh
go build -o llm-router ./cmd/llm-router/
PORT=11435 ./llm-router
```

Requires a local [Ollama](https://ollama.com) instance at `http://localhost:11434`.

---

## Running as a systemd user service

```sh
# Copy the unit file
cp deployments/spikeon-llm-router.service ~/.config/systemd/user/

# Reload and start
systemctl --user daemon-reload
systemctl --user enable --now spikeon-llm-router
```

The unit file uses `%h` (systemd's home-directory specifier) so it works for any user. By default it expects the binary at `~/Dev/spikeon-llm-router/llm-router`. Edit `WorkingDirectory` and `ExecStart` in the unit file if your path differs, then `daemon-reload` again.

To add env vars (e.g. a Gemini key), use a drop-in:

```sh
systemctl --user edit spikeon-llm-router
```

```ini
[Service]
Environment=GEMINI_API_KEY=your-key-here
```

---

## Running in Docker

### Build and run

```sh
docker build -t llm-router .
docker run -p 11435:11435 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
  llm-router
```

`host.docker.internal` resolves to the host on Mac and Windows automatically. On Linux, pass `--add-host=host.docker.internal:host-gateway` (already included in the compose file).

### Docker Compose

```sh
cd deployments
docker compose up -d
```

Edit `deployments/docker-compose.yml` to add environment variables before starting:

```yaml
environment:
  OLLAMA_BASE_URL: http://host.docker.internal:11434/v1
  GEMINI_API_KEY: your-key-here
```

For Google Drive/Gmail context, mount your credentials into the container:

```yaml
volumes:
  - ~/.config/spikeon-router:/config
environment:
  GOOGLE_TOKEN_PATH: /config/google_token.json
  GOOGLE_CREDENTIALS_PATH: /config/google_credentials.json
```

---

## Model map

| Key | Model | Use |
|---|---|---|
| snappy | [qwen2.5:0.5b](https://ollama.com/library/qwen2.5) | Trivial / factual / decomposition |
| fast | [gemma4:e2b](https://ollama.com/library/gemma4) | General chat / verification |
| memory-fast | [gemma4:e2b](https://ollama.com/library/gemma4) | Memory/preference saving (tool-capable) |
| coder | [mistral:7b](https://ollama.com/library/mistral) | Code / debug / math |
| balanced | [gemma4:e4b](https://ollama.com/library/gemma4) | Write / summarize / explain |
| thinker | [qwen3.6:latest](https://ollama.com/library/qwen3) | Deep reasoning / analysis |
| smart | [gemma4:31b](https://ollama.com/library/gemma4) | Hard / long / complex |
| orchestrator | [qwen3.6:latest](https://ollama.com/library/qwen3) | Agent orchestration (think-tokens filtered) |
| worker | [qwen3.6:latest](https://ollama.com/library/qwen3) | Heavy tool-calling subagent (think-tokens filtered) |
| gemini | [gemini-2.0-flash](https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash) | Google cloud — Drive / Sheets / Gmail |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |
| `GET` | `/v1/models` | List models |
| `GET` | `/v1/models/{id}` | Get model |
| `GET` | `/api/tags` | Ollama-style tag list |
| `GET` | `/v1/props` | Router metadata |
| `GET` | `/version` | Version string |

---

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `11435` | Listen port |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API base URL |
| `GEMINI_API_KEY` | — | Gemini API key (preferred) |
| `GEMINI_AUTH_JSON` | `~/.hermes/auth.json` | Path to JSON credential pool for Gemini key |
| `GOOGLE_TOKEN_PATH` | `~/.config/spikeon-router/google_token.json` | OAuth2 token file for Drive/Gmail |
| `GOOGLE_CREDENTIALS_PATH` | `~/.config/spikeon-router/google_credentials.json` | OAuth2 client credentials file |
| `CONV_INGEST_URL` | `http://127.0.0.1:3847/conv/log` | Conversation log ingest endpoint |

### Google OAuth setup

1. Create an OAuth2 client in [Google Cloud Console](https://console.cloud.google.com) with Drive, Sheets, and Gmail readonly scopes.
2. Download the credentials JSON and place it at `GOOGLE_CREDENTIALS_PATH`.
3. Run the OAuth flow once to produce a token at `GOOGLE_TOKEN_PATH`.

---

## Project layout

Follows [golang-standards/project-layout](https://github.com/golang-standards/project-layout).

```
cmd/llm-router/       # binary entry point
internal/
  config/             # model map, routing keywords, Caveman prompt
  router/             # prompt classification, frustration detection, decomposition, verification
  ollama/             # raw Ollama HTTP client, SSE streaming, ThinkFilter
  gemini/             # Gemini client, Google Drive/Sheets/Gmail context injection
  convlog/            # fire-and-forget conversation log HTTP client
  handlers/           # HTTP handlers — one file per endpoint group
deployments/          # systemd service unit, Docker Compose
```

---

## Dependencies

- [Ollama](https://ollama.com) — local model runtime
- [google/generative-ai-go](https://github.com/google/generative-ai-go) — Gemini API client
- [googleapis/google-api-go-client](https://github.com/googleapis/google-api-go-client) — Drive, Sheets, Gmail
- [golang.org/x/oauth2](https://pkg.go.dev/golang.org/x/oauth2) — Google OAuth2
- [google/uuid](https://github.com/google/uuid) — conversation log IDs
- [lancedb-agent-memory](https://github.com/spikeon/lancedb-agent-memory) — conversation log store (optional)
