package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/spikeon/llm-router/internal/config"
)

// openAICompatParser normalizes responses from OpenAI-shaped HTTP APIs (Ollama shim, LiteLLM, many proxies).
type openAICompatParser struct{}

func init() {
	RegisterCompletionParser(config.DefaultParserName, openAICompatParser{})
}

func (openAICompatParser) ParseCompletionBody(body []byte, httpStatus int) (Response, error) {
	body = bytes.TrimSpace(body)
	if len(body) == 0 {
		return Response{}, fmt.Errorf("empty response body")
	}
	if body[0] == '[' {
		var batch []Response
		if err := json.Unmarshal(body, &batch); err != nil {
			return Response{}, fmt.Errorf("decode response array: %w", err)
		}
		if len(batch) == 0 {
			return Response{}, fmt.Errorf("empty response array")
		}
		if len(batch[0].Choices) == 0 {
			return Response{}, fmt.Errorf("no choices in completion (body: %s)", truncateBody(body, 500))
		}
		return batch[0], nil
	}
	var out Response
	if err := json.Unmarshal(body, &out); err != nil {
		return Response{}, err
	}
	if len(out.Choices) == 0 {
		return Response{}, fmt.Errorf("no choices in completion (body: %s)", truncateBody(body, 500))
	}
	return out, nil
}

func (openAICompatParser) ParseStreamPayload(payload []byte) (StreamChunk, bool, error) {
	var chunk StreamChunk
	if err := json.Unmarshal(payload, &chunk); err != nil {
		return StreamChunk{}, false, err
	}
	return chunk, true, nil
}
