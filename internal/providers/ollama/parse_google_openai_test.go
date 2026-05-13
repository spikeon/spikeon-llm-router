package ollama

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestGoogleOpenAIParseStreamPayload_AddsMissingToolCallIndex(t *testing.T) {
	g := googleOpenAICompatParser{}
	// Gemini-style delta: tool_calls without "index" (breaks OpenAI-compatible streaming clients).
	payload := []byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"web_search","arguments":""}}]}}]}`)
	ch, ok, err := g.ParseStreamPayload(payload)
	if err != nil || !ok {
		t.Fatalf("err=%v ok=%v", err, ok)
	}
	var calls []map[string]any
	if err := json.Unmarshal(ch.Choices[0].Delta.ToolCalls, &calls); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%v", calls)
	}
	if idx, ok := calls[0]["index"]; !ok || idx != float64(0) {
		t.Fatalf("index field missing or wrong: %v", calls[0])
	}
}

func TestGoogleOpenAIParseStreamPayload_PreservesExistingIndex(t *testing.T) {
	g := googleOpenAICompatParser{}
	payload := []byte(`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"x","function":{"name":"a"}}]}}]}`)
	ch, ok, err := g.ParseStreamPayload(payload)
	if err != nil || !ok {
		t.Fatal(err)
	}
	var calls []map[string]any
	if err := json.Unmarshal(ch.Choices[0].Delta.ToolCalls, &calls); err != nil {
		t.Fatal(err)
	}
	if calls[0]["index"] != float64(1) {
		t.Fatalf("%v", calls[0])
	}
}

func TestGoogleOpenAIParseCompletionBody_MessageToolCalls(t *testing.T) {
	g := googleOpenAICompatParser{}
	body := []byte(`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"1","type":"function","function":{"name":"n"}}]}}]}`)
	out, err := g.ParseCompletionBody(body, 200)
	if err != nil {
		t.Fatal(err)
	}
	var calls []map[string]any
	raw := out.Choices[0].Message.ToolCalls
	if err := json.Unmarshal(raw, &calls); err != nil {
		t.Fatal(err)
	}
	if _, ok := calls[0]["index"]; !ok {
		t.Fatalf("expected synthetic index: %s", raw)
	}
}

func TestNormalizeGeminiToolCallsJSON_SingleObject(t *testing.T) {
	raw := []byte(`{"id":"a","type":"function","function":{"name":"x"}}`)
	out := normalizeGeminiToolCallsJSON(raw)
	if !bytes.HasPrefix(bytes.TrimSpace(out), []byte(`[`)) {
		t.Fatalf("expected array wrapper: %s", out)
	}
}
