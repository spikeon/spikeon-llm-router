package ollama

import (
	"bytes"
	"encoding/json"
)

// googleOpenAICompatParser handles Gemini's OpenAI compatibility endpoint. It can diverge from the generic
// OpenAI mapping in ParseCompletionBody / ParseStreamPayload as Google evolves the API.
type googleOpenAICompatParser struct {
	openAICompatParser
}

func init() {
	RegisterCompletionParser("google_openai", googleOpenAICompatParser{})
}

func (g googleOpenAICompatParser) ParseCompletionBody(body []byte, httpStatus int) (Response, error) {
	out, err := g.openAICompatParser.ParseCompletionBody(body, httpStatus)
	if err != nil {
		return Response{}, err
	}
	for i := range out.Choices {
		tc := out.Choices[i].Message.ToolCalls
		if len(tc) == 0 {
			continue
		}
		out.Choices[i].Message.ToolCalls = normalizeGeminiToolCallsJSON(tc)
	}
	return out, nil
}

func (g googleOpenAICompatParser) ParseStreamPayload(payload []byte) (StreamChunk, bool, error) {
	chunk, ok, err := g.openAICompatParser.ParseStreamPayload(payload)
	if err != nil || !ok {
		return chunk, ok, err
	}
	if len(chunk.Choices) > 0 {
		raw := chunk.Choices[0].Delta.ToolCalls
		if len(raw) > 0 {
			chunk.Choices[0].Delta.ToolCalls = normalizeGeminiToolCallsJSON(raw)
		}
	}
	return chunk, true, nil
}

// normalizeGeminiToolCallsJSON adds missing OpenAI-style "index" keys so streaming clients that merge
// tool_call deltas (Hermes, OpenAI SDKs) do not abort on null index. Gemini sometimes omits them.
func normalizeGeminiToolCallsJSON(raw json.RawMessage) json.RawMessage {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return raw
	}
	if raw[0] == '[' {
		var calls []map[string]any
		if err := json.Unmarshal(raw, &calls); err != nil {
			return raw
		}
		changed := false
		for i := range calls {
			if calls[i] == nil {
				continue
			}
			if _, has := calls[i]["index"]; !has {
				calls[i]["index"] = i
				changed = true
			}
		}
		if !changed {
			return raw
		}
		b, err := json.Marshal(calls)
		if err != nil {
			return raw
		}
		return b
	}
	var one map[string]any
	if err := json.Unmarshal(raw, &one); err != nil {
		return raw
	}
	if _, has := one["index"]; !has {
		one["index"] = 0
	}
	b, err := json.Marshal([]map[string]any{one})
	if err != nil {
		return raw
	}
	return b
}
