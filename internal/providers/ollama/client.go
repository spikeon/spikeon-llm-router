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

func defaultBaseURL() string {
	if u := os.Getenv("OLLAMA_BASE_URL"); u != "" {
		return strings.TrimRight(strings.TrimSpace(u), "/")
	}
	return strings.TrimRight(config.OllamaBaseURL, "/")
}

func effectiveBase(p Params) string {
	if strings.TrimSpace(p.BaseURL) != "" {
		return strings.TrimRight(strings.TrimSpace(p.BaseURL), "/")
	}
	return defaultBaseURL()
}

// ParamsForModel returns p with BaseURL, BearerToken, and ParserName set from config.Models[modelKey].
func ParamsForModel(modelKey string, p Params) Params {
	md := config.Models[modelKey]
	pk := md.ProviderKey()
	p.BaseURL = config.ProviderBaseURL(pk)
	p.ParserName = config.ParserKeyForProvider(pk)
	if md.APIKeyEnv != "" {
		p.BearerToken = strings.TrimSpace(os.Getenv(md.APIKeyEnv))
	}
	return p
}

// Msg is a single chat message.
type Msg struct {
	Role       string          `json:"role"`
	Content    string          `json:"-"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

func (m *Msg) UnmarshalJSON(data []byte) error {
	var aux struct {
		Role       string          `json:"role"`
		Content    json.RawMessage `json:"content"`
		ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
		ToolCallID string          `json:"tool_call_id,omitempty"`
		Name       string          `json:"name,omitempty"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	m.Role = aux.Role
	m.ToolCalls = aux.ToolCalls
	m.ToolCallID = aux.ToolCallID
	m.Name = aux.Name
	if len(aux.Content) == 0 || string(aux.Content) == "null" {
		m.Content = ""
		return nil
	}
	switch aux.Content[0] {
	case '"':
		return json.Unmarshal(aux.Content, &m.Content)
	case '[':
		var parts []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(aux.Content, &parts); err == nil && len(parts) > 0 {
			var b strings.Builder
			for _, part := range parts {
				b.WriteString(part.Text)
			}
			m.Content = b.String()
			return nil
		}
	}
	m.Content = strings.Trim(string(aux.Content), `"`)
	return nil
}

func (m Msg) MarshalJSON() ([]byte, error) {
	type out struct {
		Role       string          `json:"role,omitempty"`
		Content    string          `json:"content,omitempty"`
		ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
		ToolCallID string          `json:"tool_call_id,omitempty"`
		Name       string          `json:"name,omitempty"`
	}
	return json.Marshal(out{
		Role: m.Role, Content: m.Content, ToolCalls: m.ToolCalls,
		ToolCallID: m.ToolCallID, Name: m.Name,
	})
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
	Content   *string         `json:"-"`
	ToolCalls json.RawMessage `json:"tool_calls,omitempty"`
}

func (d *StreamDelta) UnmarshalJSON(data []byte) error {
	var aux struct {
		Role      string          `json:"role,omitempty"`
		Content   json.RawMessage `json:"content,omitempty"`
		ToolCalls json.RawMessage `json:"tool_calls,omitempty"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	d.Role = aux.Role
	d.ToolCalls = aux.ToolCalls
	if len(aux.Content) == 0 || string(aux.Content) == "null" {
		d.Content = nil
		return nil
	}
	switch aux.Content[0] {
	case '"':
		var s string
		if err := json.Unmarshal(aux.Content, &s); err != nil {
			return err
		}
		d.Content = &s
		return nil
	case '[':
		var parts []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(aux.Content, &parts); err == nil {
			var b strings.Builder
			for _, p := range parts {
				b.WriteString(p.Text)
			}
			s := b.String()
			d.Content = &s
			return nil
		}
	}
	s := strings.Trim(string(aux.Content), `"`)
	d.Content = &s
	return nil
}

// Params holds arguments for a chat call.
type Params struct {
	ModelName      string
	BaseURL        string // OpenAI-compat root (…/v1 or …/openai); empty => OLLAMA_BASE_URL / default local Ollama
	BearerToken    string // optional Authorization: Bearer …
	ParserName     string // CompletionParser registry key (config.ParserKeyForProvider); empty => config.DefaultParserName
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

// applyOllamaNumCtxIfSupported sets request.Options (Ollama-only num_ctx) only when the upstream
// is expected to accept it. Google AI OpenAI-compat and other strict APIs reject unknown fields like "options".
func applyOllamaNumCtxIfSupported(req *request, p Params) {
	if p.NumCtx <= 0 {
		return
	}
	if strings.TrimSpace(p.BearerToken) != "" {
		return
	}
	base := strings.ToLower(effectiveBase(p))
	switch {
	case strings.Contains(base, "generativelanguage.googleapis.com"):
		return
	case strings.Contains(base, "api.openai.com"):
		return
	case strings.Contains(base, ".openai.azure.com"):
		return
	}
	req.Options = map[string]any{"num_ctx": p.NumCtx}
}

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
	applyOllamaNumCtxIfSupported(&req, p)
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	r, err := http.NewRequest("POST", effectiveBase(p)+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	r.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(p.BearerToken) != "" {
		r.Header.Set("Authorization", "Bearer "+strings.TrimSpace(p.BearerToken))
	}
	return httpClient.Do(r)
}

func truncateBody(b []byte, max int) string {
	s := string(bytes.TrimSpace(b))
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

func readChatCompletionResponse(resp *http.Response, parser CompletionParser) (Response, error) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return Response{}, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Response{}, newUpstreamHTTPError("", resp.StatusCode, body)
	}
	out, err := parser.ParseCompletionBody(body, resp.StatusCode)
	if err != nil {
		return Response{}, fmt.Errorf("%w (body: %s)", err, truncateBody(body, 400))
	}
	return out, nil
}

// ChatSync sends a non-streaming request and returns just the assistant content.
func ChatSync(p Params) string {
	resp, err := post(p, false)
	if err != nil {
		return fmt.Sprintf("(ollama error: %v)", err)
	}
	defer resp.Body.Close()
	out, err := readChatCompletionResponse(resp, effectiveParser(p))
	if err != nil {
		return fmt.Sprintf("(ollama error: %v)", err)
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
	out, err := readChatCompletionResponse(resp, effectiveParser(p))
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// processSSELine handles one SSE line. If done is true, the stream ended with [DONE].
func processSSELine(line string, parser CompletionParser) (chunk StreamChunk, emit bool, done bool) {
	line = strings.TrimSpace(line)
	if line == "" || !strings.HasPrefix(line, "data:") {
		return StreamChunk{}, false, false
	}
	payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if payload == "[DONE]" {
		return StreamChunk{}, false, true
	}
	if payload == "" {
		return StreamChunk{}, false, false
	}
	c, ok, err := parser.ParseStreamPayload([]byte(payload))
	if err != nil {
		fmt.Printf("SSE parse error: %v payload=%q\n", err, truncateBody([]byte(payload), 200))
		return StreamChunk{}, false, false
	}
	if !ok {
		return StreamChunk{}, false, false
	}
	return c, true, false
}

// collectOpenAIStyleSSE parses OpenAI-style SSE (data: lines) from r; used by tests.
func collectOpenAIStyleSSE(r io.Reader, parser CompletionParser) ([]StreamChunk, error) {
	br := bufio.NewReader(r)
	var out []StreamChunk
	for {
		raw, err := br.ReadBytes('\n')
		chunk, emit, done := processSSELine(string(raw), parser)
		if done {
			break
		}
		if emit {
			out = append(out, chunk)
		}
		if err != nil {
			break
		}
	}
	return out, nil
}

// Stream returns a channel of StreamChunk from the Ollama SSE stream.
func Stream(p Params) (<-chan StreamChunk, error) {
	resp, err := post(p, true)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, newUpstreamHTTPError("stream", resp.StatusCode, b)
	}
	ch := make(chan StreamChunk, 32)
	parser := effectiveParser(p)
	go func() {
		defer resp.Body.Close()
		defer close(ch)
		// bufio.Scanner caps lines at 64KB by default; Gemini/Ollama SSE payloads can exceed that.
		br := bufio.NewReader(resp.Body)
		for {
			raw, err := br.ReadBytes('\n')
			chunk, emit, done := processSSELine(string(raw), parser)
			if done {
				break
			}
			if emit {
				ch <- chunk
			}
			if err != nil {
				if err != io.EOF {
					fmt.Printf("SSE read error: %v\n", err)
				}
				break
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
