package ollama

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestExtractAPIErrorMessage_geminiJSONArray(t *testing.T) {
	body := `[{"error":{"code":429,"message":"You exceeded\nyour quota for gemini-2.0-flash"}}]`
	got := extractAPIErrorMessage([]byte(body))
	want := "You exceeded your quota for gemini-2.0-flash"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestExtractAPIErrorMessage_singleObject(t *testing.T) {
	body := `{"error":{"message":"Invalid API key"}}`
	if g, w := extractAPIErrorMessage([]byte(body)), "Invalid API key"; g != w {
		t.Fatalf("got %q want %q", g, w)
	}
}

func TestNewUpstreamHTTPError_stream429(t *testing.T) {
	body := `[{"error":{"code":429,"message":"Quota exceeded"}}]`
	err := newUpstreamHTTPError("stream", 429, []byte(body))
	s := err.Error()
	if !strings.HasPrefix(s, "stream: ") {
		t.Fatalf("missing stream prefix: %q", s)
	}
	if !strings.Contains(s, "rate limit or quota exceeded") || !strings.Contains(s, "Quota exceeded") {
		t.Fatalf("got %q", s)
	}
}

func TestReadChatCompletionResponse_HTTPError_structured502(t *testing.T) {
	resp := &http.Response{
		StatusCode: 502,
		Body: io.NopCloser(strings.NewReader(
			`{"error":{"message":"bad gateway from model"}}`,
		)),
	}
	_, err := readChatCompletionResponse(resp, openAICompatParser{})
	if err == nil {
		t.Fatal("expected error")
	}
	s := err.Error()
	if !strings.Contains(s, "HTTP 502") || !strings.Contains(s, "bad gateway from model") {
		t.Fatalf("got %q", s)
	}
}
