package ollama

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestReadChatCompletionResponse_HTTPError(t *testing.T) {
	resp := &http.Response{
		StatusCode: 502,
		Body:       io.NopCloser(strings.NewReader(`{"error":"bad"}`)),
	}
	_, err := readChatCompletionResponse(resp, openAICompatParser{})
	if err == nil || !strings.Contains(err.Error(), "HTTP 502") {
		t.Fatalf("got %v", err)
	}
	// Unstructured body falls back to raw snippet
	if !strings.Contains(err.Error(), "bad") {
		t.Fatalf("got %v", err)
	}
}

func TestReadChatCompletionResponse_OK(t *testing.T) {
	resp := &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(strings.NewReader(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)),
	}
	out, err := readChatCompletionResponse(resp, openAICompatParser{})
	if err != nil {
		t.Fatal(err)
	}
	if out.Choices[0].Message.Content != "ok" {
		t.Fatal(out.Choices[0].Message.Content)
	}
}
