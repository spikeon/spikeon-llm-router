package ollama

import (
	"testing"
)

func TestApplyOllamaNumCtxIfSupported_SkipsGeminiOpenAI(t *testing.T) {
	var req request
	applyOllamaNumCtxIfSupported(&req, Params{
		NumCtx:  131072,
		BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai",
	})
	if req.Options != nil {
		t.Fatalf("google endpoint must not get options: %#v", req.Options)
	}
}

func TestApplyOllamaNumCtxIfSupported_SkipsWhenBearerSet(t *testing.T) {
	var req request
	applyOllamaNumCtxIfSupported(&req, Params{
		NumCtx:      131072,
		BaseURL:     "http://127.0.0.1:11434/v1",
		BearerToken: "secret",
	})
	if req.Options != nil {
		t.Fatal("bearer set implies non-ollama shim; skip num_ctx")
	}
}

func TestApplyOllamaNumCtxIfSupported_LocalOllama(t *testing.T) {
	var req request
	applyOllamaNumCtxIfSupported(&req, Params{
		NumCtx:  8192,
		BaseURL: "http://127.0.0.1:11434/v1",
	})
	if req.Options == nil || req.Options["num_ctx"] != 8192 {
		t.Fatalf("local ollama should get num_ctx: %#v", req.Options)
	}
}
