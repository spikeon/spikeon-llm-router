package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/spikeon/llm-router/internal/config"
	"github.com/spikeon/llm-router/internal/convlog"
	gem "github.com/spikeon/llm-router/internal/providers/gemini"
	"github.com/spikeon/llm-router/internal/providers/ollama"
	"github.com/spikeon/llm-router/internal/router"
)

// ChatRequest is the incoming OpenAI-compatible request body.
type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []ReqMessage    `json:"messages"`
	Stream         bool            `json:"stream"`
	Tools          json.RawMessage `json:"tools,omitempty"`
	ToolChoice     json.RawMessage `json:"tool_choice,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Stop           json.RawMessage `json:"stop,omitempty"`
	ResponseFormat json.RawMessage `json:"response_format,omitempty"`
	Seed           *int            `json:"seed,omitempty"`
}

// ReqMessage is one message in the incoming request.
type ReqMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

func (m ReqMessage) contentStr() string {
	if len(m.Content) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	return string(m.Content)
}

func (m ReqMessage) toMsg() ollama.Msg {
	return ollama.Msg{
		Role:       m.Role,
		Content:    m.contentStr(),
		ToolCalls:  m.ToolCalls,
		ToolCallID: m.ToolCallID,
		Name:       m.Name,
	}
}

// Chat handles POST /v1/chat/completions.
func Chat(w http.ResponseWriter, r *http.Request) {
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// collect user messages and last prompt
	var userMsgs []ReqMessage
	for _, m := range req.Messages {
		if m.Role == "user" {
			userMsgs = append(userMsgs, m)
		}
	}
	lastPrompt := ""
	if len(userMsgs) > 0 {
		lastPrompt = userMsgs[len(userMsgs)-1].contentStr()
	}

	// orchestrator / worker bypass: full tool-call support, think-token filtering
	if req.Model == "orchestrator" || req.Model == "worker" {
		handleAgentModel(w, r, req, lastPrompt)
		return
	}

	t0 := time.Now()

	// frustration detection
	frustrated := router.IsFrustrated(lastPrompt)
	modelKey := ""
	if frustrated {
		modelKey = "smart"
		// try to re-run last non-frustrated prompt
		for i := len(userMsgs) - 2; i >= 0; i-- {
			if !router.IsFrustrated(userMsgs[i].contentStr()) {
				lastPrompt = userMsgs[i].contentStr()
				fmt.Printf("→ FRUSTRATED — re-running: '%s...' with smart\n", trunc(lastPrompt, 50))
				break
			}
		}
		if modelKey == "" {
			fmt.Println("→ FRUSTRATED — no prior question, using smart on current")
		}
	} else {
		if req.Model == "auto" {
			modelKey = router.Classify(lastPrompt)
		} else {
			modelKey = req.Model
		}
	}

	// build message list with system prompt injection
	msgs := buildMessages(req)

	// replace last user message if re-running a prior prompt
	if frustrated {
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Role == "user" {
				msgs[i].Content = lastPrompt
				break
			}
		}
	}

	// task decomposition
	inToolCont := len(req.Messages) > 0 && req.Messages[len(req.Messages)-1].Role == "tool"
	if !frustrated && req.Model == "auto" && !inToolCont && router.ShouldDecompose(lastPrompt) {
		tasks := router.Decompose(lastPrompt)
		if len(tasks) > 1 {
			fmt.Printf("\n%s\nDECOMPOSE → %d tasks\n", strings.Repeat("=", 60), len(tasks))
			for i, t := range tasks {
				fmt.Printf("  %d. %s\n", i+1, trunc(t, 90))
			}
			fmt.Println(strings.Repeat("=", 60))
			result := runDecomposed(tasks, msgs, req)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(result)
			return
		}
	}

	fmt.Printf("→ routing to: %s (%s)\n", modelKey, config.Models[modelKey].Name)

	// Gemini cloud path
	if modelKey == "gemini" {
		system, history := extractSystemAndHistory(msgs)
		if req.Stream {
			handleGeminiStream(w, lastPrompt, system, history, frustrated, t0)
		} else {
			handleGeminiSync(w, lastPrompt, system, history, frustrated, t0)
		}
		return
	}

	// Ollama local path
	modelName := config.Models[modelKey].Name
	params := ollama.ParamsForModel(modelKey, ollama.Params{
		ModelName:      modelName,
		Messages:       msgs,
		Tools:          req.Tools,
		ToolChoice:     req.ToolChoice,
		Temperature:    req.Temperature,
		MaxTokens:      req.MaxTokens,
		TopP:           req.TopP,
		Stop:           req.Stop,
		ResponseFormat: req.ResponseFormat,
		Seed:           req.Seed,
	})
	if req.Stream {
		handleOllamaStream(w, params, modelKey, lastPrompt, frustrated, t0)
	} else {
		handleOllamaSync(w, params, modelKey, modelName, lastPrompt, frustrated, t0)
	}
}

// handleAgentModel handles the orchestrator/worker bypass path.
func handleAgentModel(w http.ResponseWriter, r *http.Request, req ChatRequest, lastPrompt string) {
	key := req.Model
	modelName := config.Models[key].Name
	filterThink := strings.Contains(strings.ToLower(modelName), "qwen3")

	msgs := make([]ollama.Msg, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = m.toMsg()
	}

	// inject tool directory into system prompt
	var toolNames []string
	if len(req.Tools) > 0 {
		var tools []map[string]any
		if err := json.Unmarshal(req.Tools, &tools); err == nil {
			for _, t := range tools {
				if fn, ok := t["function"].(map[string]any); ok {
					if name, ok := fn["name"].(string); ok {
						toolNames = append(toolNames, name)
					}
				}
			}
		}
	}
	lastRole := ""
	if len(msgs) > 0 {
		lastRole = msgs[len(msgs)-1].Role
	}
	fmt.Printf("→ %s (%s) stream=%v tools=%d last_role=%s\n", key, modelName, req.Stream, len(toolNames), lastRole)
	if len(msgs) > 0 {
		fmt.Printf("  last_msg: %q\n", trunc(msgs[len(msgs)-1].Content, 120))
	}

	if len(toolNames) > 0 && lastRole == "user" {
		byPrefix := map[string][]string{}
		for _, t := range toolNames {
			if strings.HasPrefix(t, "mcp_") {
				parts := strings.SplitN(t, "_", 3)
				prefix := "other"
				if len(parts) >= 2 {
					prefix = parts[1]
				}
				byPrefix[prefix] = append(byPrefix[prefix], t)
			}
		}
		lines := []string{"AVAILABLE MCP TOOLS (use exact names):"}
		for prefix, tools := range byPrefix {
			suffix := ""
			if len(tools) > 6 {
				tools = tools[:6]
				suffix = "..."
			}
			lines = append(lines, fmt.Sprintf("  %s: %s%s", prefix, strings.Join(tools, ", "), suffix))
		}
		toolDir := strings.Join(lines, "\n")
		injected := false
		for i, m := range msgs {
			if m.Role == "system" {
				msgs[i].Content = m.Content + "\n\n" + toolDir
				injected = true
				break
			}
		}
		if !injected {
			msgs = append([]ollama.Msg{{Role: "system", Content: toolDir}}, msgs...)
		}
	}

	params := ollama.ParamsForModel(key, ollama.Params{
		ModelName:      modelName,
		Messages:       msgs,
		Tools:          req.Tools,
		ToolChoice:     req.ToolChoice,
		Temperature:    req.Temperature,
		MaxTokens:      req.MaxTokens,
		TopP:           req.TopP,
		Stop:           req.Stop,
		ResponseFormat: req.ResponseFormat,
		Seed:           req.Seed,
		NumCtx:         131072,
	})

	if req.Stream {
		ollama.SSEHeaders(w)
		var filter *ollama.ThinkFilter
		if filterThink {
			filter = &ollama.ThinkFilter{}
		}
		chunks, err := ollama.Stream(params)
		if err != nil {
			fmt.Printf("stream upstream error: %v\n", err)
			fmt.Fprintf(w, "data: {\"error\":%q}\n\n", err.Error())
			return
		}
		t0 := time.Now()
		flusher, _ := w.(http.Flusher)
		fullResp := ollama.ProxySSE(w, flusher, chunks, key, filter)
		convlog.LogAsync(convlog.Entry{
			Prompt: lastPrompt, Response: fullResp,
			ModelKey: key, ModelName: modelName,
			LatencyMS: float64(time.Since(t0).Milliseconds()),
		})
		return
	}

	// non-streaming
	t0 := time.Now()
	resp, err := ollama.Chat(params)
	if err != nil {
		fmt.Printf("chat upstream error: %v\n", err)
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	choice := resp.Choices[0]
	content := choice.Message.Content
	if filterThink {
		content = ollama.StripThink(content)
	}
	hasTC := len(choice.Message.ToolCalls) > 0
	fmt.Printf("  finish=%s content_len=%d tool_calls=%v\n", choice.FinishReason, len(content), hasTC)

	convlog.LogAsync(convlog.Entry{
		Prompt: lastPrompt, Response: content,
		ModelKey: key, ModelName: modelName,
		LatencyMS:    float64(time.Since(t0).Milliseconds()),
		FinishReason: choice.FinishReason,
		HasToolCalls: hasTC,
	})

	msg := map[string]any{"role": "assistant", "content": content}
	if hasTC {
		msg["tool_calls"] = json.RawMessage(choice.Message.ToolCalls)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": resp.ID, "object": "chat.completion", "model": key,
		"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": choice.FinishReason}},
	})
}

// runDecomposed executes tasks sequentially, forwarding accumulated context.
func runDecomposed(tasks []string, baseMessages []ollama.Msg, req ChatRequest) map[string]any {
	prefix := baseMessages[:len(baseMessages)-1]
	var taskList []string
	var responses []string
	wallStart := time.Now()

	for i, task := range tasks {
		taskMsgs := make([]ollama.Msg, len(prefix))
		copy(taskMsgs, prefix)
		for j := range taskList {
			taskMsgs = append(taskMsgs, ollama.Msg{Role: "user", Content: taskList[j]})
			taskMsgs = append(taskMsgs, ollama.Msg{Role: "assistant", Content: responses[j]})
		}
		taskMsgs = append(taskMsgs, ollama.Msg{Role: "user", Content: task})

		modelKey := router.Classify(task)
		modelName := config.Models[modelKey].Name
		fmt.Printf("\n[task %d/%d] → %s (%s)\n  prompt: %s\n", i+1, len(tasks), modelKey, modelName, trunc(task, 80))
		t0 := time.Now()

		if modelKey == "gemini" {
			system, history := extractSystemAndHistory(taskMsgs)
			text, err := gem.Chat(task, system, history)
			if err != nil {
				text = fmt.Sprintf("(gemini error: %v)", err)
			}
			elapsed := time.Since(t0)
			fmt.Printf("  ✓ done in %.1fs\n", elapsed.Seconds())
			convlog.LogAsync(convlog.Entry{
				Prompt: task, Response: text, ModelKey: "gemini",
				ModelName:  config.Models["gemini"].Name,
				LatencyMS:  float64(elapsed.Milliseconds()),
				Decomposed: true, TaskIndex: i + 1, TotalTasks: len(tasks),
			})
			taskList = append(taskList, task)
			responses = append(responses, text)
			continue
		}

		resp, err := ollama.Chat(ollama.ParamsForModel(modelKey, ollama.Params{
			ModelName: modelName, Messages: taskMsgs,
			Tools: req.Tools, ToolChoice: req.ToolChoice,
		}))
		elapsed := time.Since(t0)
		if err != nil {
			fmt.Printf("  ✗ error: %v\n", err)
			taskList = append(taskList, task)
			responses = append(responses, fmt.Sprintf("(error: %v)", err))
			continue
		}

		choice := resp.Choices[0]
		hasTC := len(choice.Message.ToolCalls) > 0
		if hasTC {
			fmt.Println("  ↩ tool_call returned — surfacing to client")
			convlog.LogAsync(convlog.Entry{
				Prompt: task, Response: choice.Message.Content,
				ModelKey: modelKey, ModelName: modelName,
				LatencyMS: float64(elapsed.Milliseconds()),
				Decomposed: true, TaskIndex: i + 1, TotalTasks: len(tasks),
				FinishReason: choice.FinishReason, HasToolCalls: true,
			})
			msg := map[string]any{"role": "assistant", "content": choice.Message.Content, "tool_calls": json.RawMessage(choice.Message.ToolCalls)}
			return map[string]any{
				"id": resp.ID, "object": "chat.completion", "model": modelKey,
				"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": choice.FinishReason}},
			}
		}

		fmt.Printf("  ✓ done in %.1fs\n", elapsed.Seconds())
		convlog.LogAsync(convlog.Entry{
			Prompt: task, Response: choice.Message.Content,
			ModelKey: modelKey, ModelName: modelName,
			LatencyMS: float64(elapsed.Milliseconds()),
			Decomposed: true, TaskIndex: i + 1, TotalTasks: len(tasks),
			FinishReason: choice.FinishReason,
		})
		taskList = append(taskList, task)
		responses = append(responses, choice.Message.Content)
	}

	// verify step
	originalPrompt := ""
	for i := len(baseMessages) - 1; i >= 0; i-- {
		if baseMessages[i].Role == "user" {
			originalPrompt = baseMessages[i].Content
			break
		}
	}
	fmt.Println("\n[verify] comparing responses to original request...")
	t0 := time.Now()
	verification := router.VerifyCompletion(originalPrompt, taskList, responses)
	fmt.Printf("  ✓ done in %.1fs\n", time.Since(t0).Seconds())
	for _, line := range strings.Split(verification, "\n") {
		if strings.Contains(line, "VERDICT") {
			fmt.Printf("  %s\n", line)
		}
	}

	fmt.Printf("\n%s\nDECOMPOSE complete — %d tasks in %.1fs total\n%s\n\n",
		strings.Repeat("=", 60), len(taskList), time.Since(wallStart).Seconds(), strings.Repeat("=", 60))

	final := ""
	if len(responses) > 0 {
		final = responses[len(responses)-1]
	}
	final += "\n\n---\n" + verification

	return map[string]any{
		"id": "decomposed-response", "object": "chat.completion", "model": "auto-decomposed",
		"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": final}, "finish_reason": "stop"}},
	}
}

func handleGeminiSync(w http.ResponseWriter, prompt, system string, history []ollama.Msg, frustrated bool, t0 time.Time) {
	text, err := gem.Chat(prompt, system, history)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	convlog.LogAsync(convlog.Entry{
		Prompt: prompt, Response: text,
		ModelKey: "gemini", ModelName: config.Models["gemini"].Name,
		LatencyMS: float64(time.Since(t0).Milliseconds()),
		Frustrated: frustrated, FinishReason: "stop",
	})
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": "gemini-response", "object": "chat.completion", "model": "gemini",
		"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": text}, "finish_reason": "stop"}},
	})
}

func handleGeminiStream(w http.ResponseWriter, prompt, system string, history []ollama.Msg, frustrated bool, t0 time.Time) {
	ollama.SSEHeaders(w)
	flusher, _ := w.(http.Flusher)
	ch, err := gem.Stream(prompt, system, history)
	if err != nil {
		fmt.Fprintf(w, "data: {\"error\":%q}\n\n", err.Error())
		return
	}
	var full strings.Builder
	for text := range ch {
		full.WriteString(text)
		out := map[string]any{
			"id": "gemini-stream", "object": "chat.completion.chunk", "model": "gemini",
			"choices": []map[string]any{{"index": 0, "delta": map[string]any{"role": "assistant", "content": text}, "finish_reason": nil}},
		}
		b, _ := json.Marshal(out)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if flusher != nil {
			flusher.Flush()
		}
	}
	convlog.LogAsync(convlog.Entry{
		Prompt: prompt, Response: full.String(),
		ModelKey: "gemini", ModelName: config.Models["gemini"].Name,
		LatencyMS: float64(time.Since(t0).Milliseconds()),
		Frustrated: frustrated, FinishReason: "stop",
	})
	io.WriteString(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

func handleOllamaSync(w http.ResponseWriter, params ollama.Params, modelKey, modelName, prompt string, frustrated bool, t0 time.Time) {
	resp, err := ollama.Chat(params)
	if err != nil {
		fmt.Printf("chat upstream error: %v\n", err)
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	choice := resp.Choices[0]
	hasTC := len(choice.Message.ToolCalls) > 0
	convlog.LogAsync(convlog.Entry{
		Prompt: prompt, Response: choice.Message.Content,
		ModelKey: modelKey, ModelName: modelName,
		LatencyMS:    float64(time.Since(t0).Milliseconds()),
		Frustrated:   frustrated,
		FinishReason: choice.FinishReason,
		HasToolCalls: hasTC,
	})
	msg := map[string]any{"role": "assistant", "content": choice.Message.Content}
	if hasTC {
		msg["tool_calls"] = json.RawMessage(choice.Message.ToolCalls)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": resp.ID, "object": "chat.completion", "model": modelKey,
		"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": choice.FinishReason}},
	})
}

func handleOllamaStream(w http.ResponseWriter, params ollama.Params, modelKey, prompt string, frustrated bool, t0 time.Time) {
	ollama.SSEHeaders(w)
	chunks, err := ollama.Stream(params)
	if err != nil {
		fmt.Fprintf(w, "data: {\"error\":%q}\n\n", err.Error())
		return
	}
	flusher, _ := w.(http.Flusher)
	fullResp := ollama.ProxySSE(w, flusher, chunks, modelKey, nil)
	convlog.LogAsync(convlog.Entry{
		Prompt: prompt, Response: fullResp,
		ModelKey: modelKey, ModelName: params.ModelName,
		LatencyMS: float64(time.Since(t0).Milliseconds()),
		Frustrated: frustrated, FinishReason: "stop",
	})
}

// buildMessages converts request messages to ollama.Msg, injecting the system prompt.
func buildMessages(req ChatRequest) []ollama.Msg {
	hasSystem := false
	for _, m := range req.Messages {
		if m.Role == "system" {
			hasSystem = true
			break
		}
	}
	msgs := make([]ollama.Msg, 0, len(req.Messages)+1)
	if !hasSystem {
		msgs = append(msgs, ollama.Msg{Role: "system", Content: config.CavemanSystemPrompt})
	}
	for _, m := range req.Messages {
		msg := m.toMsg()
		if m.Role == "system" {
			msg.Content = m.contentStr() + "\n\nADDITIONAL INSTRUCTIONS: " + config.CavemanSystemPrompt
		}
		msgs = append(msgs, msg)
	}
	return msgs
}

// extractSystemAndHistory splits a message list into the system prompt string
// and the conversation history (user/assistant only, excluding the last user message).
func extractSystemAndHistory(msgs []ollama.Msg) (system string, history []ollama.Msg) {
	system = config.CavemanSystemPrompt
	for _, m := range msgs {
		if m.Role == "system" {
			system = m.Content
		}
	}
	for _, m := range msgs {
		if m.Role == "user" || m.Role == "assistant" {
			history = append(history, m)
		}
	}
	// drop the last user message (it's the prompt, sent separately)
	if len(history) > 0 && history[len(history)-1].Role == "user" {
		history = history[:len(history)-1]
	}
	return
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
