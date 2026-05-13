package router

import (
	"fmt"
	"strings"

	"github.com/spikeon/llm-router/internal/config"
	"github.com/spikeon/llm-router/internal/providers/ollama"
)

var verifySystem = `You are a completion verifier. Compare the original request against the task responses below.
Output format (be concise):
✓ Completed: [bullet list of what was done]
✗ Missed: [anything not addressed, or "nothing"]
VERDICT: Complete / Incomplete`

// VerifyCompletion calls the fast model to validate decomposed task results.
func VerifyCompletion(originalPrompt string, tasks []string, responses []string) string {
	parts := []string{fmt.Sprintf("ORIGINAL REQUEST:\n%s\n\nTASK RESPONSES:", originalPrompt)}
	for i, task := range tasks {
		resp := ""
		if i < len(responses) {
			resp = responses[i]
			if len(resp) > 600 {
				resp = resp[:600]
			}
		}
		parts = append(parts, fmt.Sprintf("\nTask %d — %s\n%s", i+1, task, resp))
	}
	return ollama.ChatSync(ollama.ParamsForModel("fast", ollama.Params{
		ModelName: config.Models["fast"].Name,
		Messages: []ollama.Msg{
			{Role: "system", Content: verifySystem},
			{Role: "user", Content: strings.Join(parts, "")},
		},
	}))
}
