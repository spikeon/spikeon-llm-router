package ollama

import (
	"testing"
)

func TestOpenAICompatParseCompletionBody_Object(t *testing.T) {
	p := openAICompatParser{}
	body := []byte(`{"id":"1","object":"chat.completion","model":"x","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
	out, err := p.ParseCompletionBody(body, 200)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Choices) != 1 || out.Choices[0].Message.Content != "hi" {
		t.Fatalf("got %+v", out)
	}
}

func TestOpenAICompatParseCompletionBody_ArrayRoot(t *testing.T) {
	p := openAICompatParser{}
	body := []byte(`[{"id":"1","choices":[{"index":0,"message":{"role":"assistant","content":"yo"},"finish_reason":"stop"}]}]`)
	out, err := p.ParseCompletionBody(body, 200)
	if err != nil {
		t.Fatal(err)
	}
	if out.Choices[0].Message.Content != "yo" {
		t.Fatalf("got %+v", out.Choices[0].Message)
	}
}

func TestOpenAICompatParseCompletionBody_MessageContentParts(t *testing.T) {
	p := openAICompatParser{}
	body := []byte(`{"choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"aa"},{"type":"text","text":"bb"}]}}]}`)
	out, err := p.ParseCompletionBody(body, 200)
	if err != nil {
		t.Fatal(err)
	}
	if got := out.Choices[0].Message.Content; got != "aabb" {
		t.Fatalf("content %q", got)
	}
}

func TestOpenAICompatParseCompletionBody_NoChoices(t *testing.T) {
	p := openAICompatParser{}
	_, err := p.ParseCompletionBody([]byte(`{"choices":[]}`), 200)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestOpenAICompatParseStreamPayload(t *testing.T) {
	p := openAICompatParser{}
	payload := []byte(`{"id":"c1","choices":[{"index":0,"delta":{"content":"z"}}]}`)
	ch, ok, err := p.ParseStreamPayload(payload)
	if err != nil || !ok {
		t.Fatalf("err=%v ok=%v", err, ok)
	}
	if *ch.Choices[0].Delta.Content != "z" {
		t.Fatal(*ch.Choices[0].Delta.Content)
	}
}
