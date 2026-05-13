package ollama

import (
	"strings"
	"sync"

	"github.com/spikeon/llm-router/internal/config"
)

// CompletionParser adapts provider-specific chat-completions JSON (non-streaming and SSE payloads)
// into router-internal types. Register implementations with RegisterCompletionParser; reference them
// from config.ProviderDef.Parser or rely on DefaultParserName ("openai_compat").
type CompletionParser interface {
	// ParseCompletionBody runs after HTTP status is verified as 2xx. Body is the raw response entity.
	ParseCompletionBody(body []byte, httpStatus int) (Response, error)
	// ParseStreamPayload decodes one SSE "data: …" line (excluding the "[DONE]" sentinel).
	// ok=false means skip emitting this event (malformed line); err non-nil aborts the stream.
	ParseStreamPayload(payload []byte) (chunk StreamChunk, ok bool, err error)
}

var (
	completionParsersMu sync.RWMutex
	completionParsers   = map[string]CompletionParser{}
)

// RegisterCompletionParser registers a parser by name (e.g. from config.ProviderDef.Parser).
// Safe to call from init() in this package or from another package linked via blank import in main.
func RegisterCompletionParser(name string, p CompletionParser) {
	name = strings.TrimSpace(name)
	if name == "" || p == nil {
		return
	}
	completionParsersMu.Lock()
	defer completionParsersMu.Unlock()
	completionParsers[name] = p
}

func completionParserNamed(name string) CompletionParser {
	name = strings.TrimSpace(name)
	if name == "" {
		name = config.DefaultParserName
	}
	completionParsersMu.RLock()
	p, ok := completionParsers[name]
	completionParsersMu.RUnlock()
	if ok {
		return p
	}
	completionParsersMu.RLock()
	fallback, ok := completionParsers[config.DefaultParserName]
	completionParsersMu.RUnlock()
	if ok {
		return fallback
	}
	return openAICompatParser{}
}

func effectiveParser(p Params) CompletionParser {
	name := strings.TrimSpace(p.ParserName)
	if name == "" {
		name = config.DefaultParserName
	}
	return completionParserNamed(name)
}
