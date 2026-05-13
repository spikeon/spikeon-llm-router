package router

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"github.com/spikeon/llm-router/internal/config"
	"github.com/spikeon/llm-router/internal/ollama"
)

var (
	rePersonalInfo = regexp.MustCompile(
		`(?i)\b(my|your) .+ is\b` +
			`|\bi (am|prefer|like|hate|love|use|have)\b` +
			`|\bwhat('?s| is| are) (my|your)\b` +
			`|\bdo you (know|remember)\b` +
			`|\bwhat did i\b` +
			`|\btell me (about )?(my|your)\b`,
	)
	reDomain      = regexp.MustCompile(`(?i)\b\w+\.(com|org|net|io|dev|ai)\b`)
	reWebRef      = regexp.MustCompile(`(?i)\baccording to\b|\blook up\b|\bsearch for\b|\bfind on\b|\bon the web\b|\bonline\b`)
	reTaskLine    = regexp.MustCompile(`(?m)^\d+[.)]\s+(.+)`)
	reSentenceSep = regexp.MustCompile(`[.!?]`)
)

var decomposeConnectors = []string{
	"and then", "and also", "and make", "and change", "and move",
	"and set", "and update", "and add", "and remove", "and fix",
	"as well as", "additionally", "furthermore", "after that",
	", and ", " then ", " also ",
}

const decomposeTokenMin = 20

func countTokens(text string) float64 {
	return float64(len(strings.Fields(text))) * 1.3
}

func containsAny(s string, keywords []string) bool {
	lower := strings.ToLower(s)
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

func wordBoundaryMatch(text string, keywords []string) bool {
	lower := strings.ToLower(text)
	for _, kw := range keywords {
		re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(kw) + `\b`)
		if re.MatchString(lower) {
			return true
		}
	}
	return false
}

// Classify returns the routing model key for a prompt.
func Classify(prompt string) string {
	lower := strings.ToLower(prompt)
	tokens := countTokens(prompt)

	if containsAny(lower, config.FinanceKeywords) || containsAny(lower, config.GoogleKeywords) {
		return "gemini"
	}
	if wordBoundaryMatch(lower, config.ReasoningKeywords) {
		return "thinker"
	}
	if rePersonalInfo.MatchString(lower) || containsAny(lower, config.MemoryKeywords) {
		return "memory-fast"
	}
	if reDomain.MatchString(lower) || reWebRef.MatchString(lower) {
		return "balanced"
	}
	if containsAny(lower, config.CodeKeywords) {
		return "coder"
	}
	if tokens > config.TokenThresholdLong {
		return "smart"
	}
	if containsAny(lower, config.BalancedKeywords) {
		return "balanced"
	}
	return "fast"
}

// IsFrustrated returns true if the prompt contains swearing or is mostly uppercase.
func IsFrustrated(prompt string) bool {
	if containsAny(strings.ToLower(prompt), config.SwearKeywords) {
		return true
	}
	var alphas []rune
	upper := 0
	for _, r := range prompt {
		if unicode.IsLetter(r) {
			alphas = append(alphas, r)
			if unicode.IsUpper(r) {
				upper++
			}
		}
	}
	return len(alphas) > 3 && float64(upper)/float64(len(alphas)) > 0.5
}

// ShouldDecompose returns true if the prompt looks like multiple chained tasks.
func ShouldDecompose(prompt string) bool {
	if countTokens(prompt) < decomposeTokenMin {
		return false
	}
	lower := strings.ToLower(prompt)
	hits := 0
	for _, c := range decomposeConnectors {
		hits += strings.Count(lower, c)
	}
	sentences := 0
	for _, s := range reSentenceSep.Split(prompt, -1) {
		if len(strings.TrimSpace(s)) > 5 {
			sentences++
		}
	}
	return hits >= 2 || (hits >= 1 && sentences >= 2) || sentences >= 3
}

var decomposeSystem = `Break the user's request into a numbered list of atomic, sequential tasks.
Rules:
- One clear action per task
- Logical order of operations (discover before modifying)
- Use $variables for values found in earlier steps
- Output ONLY the numbered list, nothing else`

// Decompose calls the snappy model to break a prompt into subtasks.
func Decompose(prompt string) []string {
	body := ollama.ChatSync(ollama.Params{
		ModelName: config.Models["snappy"].Name,
		Messages: []ollama.Msg{
			{Role: "system", Content: decomposeSystem},
			{Role: "user", Content: prompt},
		},
	})
	var tasks []string
	for _, m := range reTaskLine.FindAllStringSubmatch(body, -1) {
		if t := strings.TrimSpace(m[1]); t != "" {
			tasks = append(tasks, t)
		}
	}
	if len(tasks) > 1 {
		return tasks
	}
	return []string{prompt}
}

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
	return ollama.ChatSync(ollama.Params{
		ModelName: config.Models["fast"].Name,
		Messages: []ollama.Msg{
			{Role: "system", Content: verifySystem},
			{Role: "user", Content: strings.Join(parts, "")},
		},
	})
}
