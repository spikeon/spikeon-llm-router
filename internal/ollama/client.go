package ollama

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/spikeon/llm-router/internal/config"
)

func baseURL() string {
	if u := os.Getenv("OLLAMA_BASE_URL"); u != "" {
		return u
	}
	return config.OllamaBaseURL
}

// Msg is a single chat message.
type Msg struct {
	Role       string          `json:"role"`
	Content    string          `json:"content"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

type request struct {
	Model          string          `json:"model"`
	Messages       []Msg           `json:"messages"`
	Stream         bool            `json:"stream"`
	Tools          json.RawMessage `json:"tools,omitempty"`
	ToolChoice     json.RawMessage `json:"tool_choice,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Stop           json.RawMessage `json:"stop,omitempty"`
	ResponseFormat json.RawMessage `json:"response_format,omitempty"`
	Seed           *int            `json:"seed,omitempty"`
	Options        map[string]any  `json:"options,omitempty"`
}

// Response is a non-streaming chat completion response.
type Response struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}

// Choice is one item in Response.Choices.
type Choice struct {
	Index        int    `json:"index"`
	Message      Msg    `json:"message"`
	FinishReason string `json:"finish_reason"`
}

// StreamChunk is one SSE chunk from the Ollama streaming API.
type StreamChunk struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Model   string          `json:"model"`
	Choices []StreamChoice  `json:"choices"`
}

// StreamChoice is one element of Choices in a stream chunk.
type StreamChoice struct {
	Index        int          `json:"index"`
	Delta        StreamDelta  `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

// StreamDelta carries the incremental content in a stream chunk.
type StreamDelta struct {
	Role      string          `json:"role,omitempty"`
	Content   *string         `json:"content,omitempty"`
	ToolCalls json.RawMessage `json:"tool_calls,omitempty"`
}

// Params holds arguments for a chat call.
type Params struct {
	ModelName      string
	Messages       []Msg
	Tools          json.RawMessage
	ToolChoice     json.RawMessage
	Temperature    *float64
	MaxTokens      *int
	TopP           *float64
	Stop           json.RawMessage
	ResponseFormat json.RawMessage
	Seed           *int
	NumCtx         int
}

var httpClient = &http.Client{}

func post(p Params, stream bool) (*http.Response, error) {
	req := request{
		Model:          p.ModelName,
		Messages:       p.Messages,
		Stream:         stream,
		Tools:          p.Tools,
		ToolChoice:     p.ToolChoice,
		Temperature:    p.Temperature,
		MaxTokens:      p.MaxTokens,
		TopP:           p.TopP,
		Stop:           p.Stop,
		ResponseFormat: p.ResponseFormat,
		Seed:           p.Seed,
	}
	if p.NumCtx > 0 {
		req.Options = map[string]any{"num_ctx": p.NumCtx}
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	r, err := http.NewRequest("POST", baseURL()+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	r.Header.Set("Content-Type", "application/json")
	return httpClient.Do(r)
}

// ChatSync sends a non-streaming request and returns just the assistant content.
func ChatSync(p Params) string {
	resp, err := post(p, false)
	if err != nil {
		return fmt.Sprintf("(ollama error: %v)", err)
	}
	defer resp.Body.Close()
	var out Response
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Choices) == 0 {
		return ""
	}
	return out.Choices[0].Message.Content
}

// Chat sends a non-streaming request and returns the full Response.
func Chat(p Params) (*Response, error) {
	resp, err := post(p, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out Response
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Stream returns a channel of StreamChunk from the Ollama SSE stream.
func Stream(p Params) (<-chan StreamChunk, error) {
	resp, err := post(p, true)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamChunk, 32)
	go func() {
		defer resp.Body.Close()
		defer close(ch)
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := line[6:]
			if payload == "[DONE]" {
				break
			}
			var chunk StreamChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err == nil {
				ch <- chunk
			}
		}
	}()
	return ch, nil
}

// ProxySSE copies SSE chunks from an Ollama stream to the ResponseWriter,
// optionally filtering think tokens. Returns the accumulated full response text.
func ProxySSE(w io.Writer, flusher http.Flusher, chunks <-chan StreamChunk, modelKey string, filter *ThinkFilter) string {
	var full strings.Builder
	for chunk := range chunks {
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		delta := choice.Delta

		outDelta := map[string]any{"role": "assistant"}
		if delta.Content != nil {
			text := *delta.Content
			if filter != nil {
				text = filter.Write(text)
			}
			if text != "" {
				outDelta["content"] = text
				full.WriteString(text)
			}
		}
		if len(delta.ToolCalls) > 0 {
			outDelta["tool_calls"] = json.RawMessage(delta.ToolCalls)
		}

		out := map[string]any{
			"id":     chunk.ID,
			"object": "chat.completion.chunk",
			"model":  modelKey,
			"choices": []map[string]any{{
				"index":         choice.Index,
				"delta":         outDelta,
				"finish_reason": choice.FinishReason,
			}},
		}
		b, _ := json.Marshal(out)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if flusher != nil {
			flusher.Flush()
		}
	}
	if filter != nil {
		if tail := filter.Flush(); tail != "" {
			out := map[string]any{
				"id":     "flush",
				"object": "chat.completion.chunk",
				"model":  modelKey,
				"choices": []map[string]any{{
					"index": 0, "delta": map[string]any{"content": tail}, "finish_reason": nil,
				}},
			}
			b, _ := json.Marshal(out)
			fmt.Fprintf(w, "data: %s\n\n", b)
			full.WriteString(tail)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}
	io.WriteString(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
	return full.String()
}

// SSEHeaders sets headers required for Server-Sent Events.
func SSEHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
}
