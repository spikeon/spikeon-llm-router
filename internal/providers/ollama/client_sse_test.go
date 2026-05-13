package ollama

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/spikeon/llm-router/internal/config"
)

func TestProcessSSELine_Done(t *testing.T) {
	_, emit, done := processSSELine("data: [DONE]", openAICompatParser{})
	if emit || !done {
		t.Fatalf("emit=%v done=%v", emit, done)
	}
}

func TestProcessSSELine_DataPrefixWithoutSpace(t *testing.T) {
	chunk, emit, done := processSSELine(`data:{"choices":[{"delta":{"content":"a"}}]}`, openAICompatParser{})
	if done || !emit || *chunk.Choices[0].Delta.Content != "a" {
		t.Fatalf("done=%v emit=%v chunk=%+v", done, emit, chunk)
	}
}

func TestCollectOpenAIStyleSSE_MultiChunkAndDone(t *testing.T) {
	sse := "event: ping\ndata: {\"choices\":[{\"delta\":{\"content\":\"h\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"i\"}}]}\n\ndata: [DONE]\n"
	chunks, err := collectOpenAIStyleSSE(strings.NewReader(sse), openAICompatParser{})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 {
		t.Fatalf("len=%d", len(chunks))
	}
	if *chunks[0].Choices[0].Delta.Content != "h" || *chunks[1].Choices[0].Delta.Content != "i" {
		t.Fatal("content mismatch")
	}
}

func TestCollectOpenAIStyleSSE_LongLine_NoScannerLimit(t *testing.T) {
	// Regression: bufio.Scanner defaulted to 64KB max line — large SSE payloads must still parse.
	big := strings.Repeat("X", 150_000)
	chunkObj := map[string]any{
		"choices": []map[string]any{{
			"index": 0,
			"delta": map[string]any{"content": big},
		}},
	}
	payload, err := json.Marshal(chunkObj)
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	b.WriteString("data: ")
	b.Write(payload)
	b.WriteByte('\n')
	b.WriteString("data: [DONE]\n")
	chunks, err := collectOpenAIStyleSSE(strings.NewReader(b.String()), openAICompatParser{})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	got := *chunks[0].Choices[0].Delta.Content
	if len(got) != len(big) || got != big {
		t.Fatalf("length want %d got %d", len(big), len(got))
	}
}

func TestCompletionParserRegistry_HasBuiltins(t *testing.T) {
	_ = completionParserNamed(config.DefaultParserName) // panics if init did not register
	_ = completionParserNamed("google_openai")
}
