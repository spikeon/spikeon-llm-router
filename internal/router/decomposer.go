package router

import (
	"regexp"
	"strings"

	"github.com/spikeon/llm-router/internal/config"
	"github.com/spikeon/llm-router/internal/providers/ollama"
)

var (
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

var decomposeSystem = `Break the user's request into a numbered list of atomic, sequential tasks.
Rules:
- One clear action per task
- Logical order of operations (discover before modifying)
- Use $variables for values found in earlier steps
- Output ONLY the numbered list, nothing else`

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

// Decompose calls the snappy model to break a prompt into subtasks.
func Decompose(prompt string) []string {
	body := ollama.ChatSync(ollama.ParamsForModel("snappy", ollama.Params{
		ModelName: config.Models["snappy"].Name,
		Messages: []ollama.Msg{
			{Role: "system", Content: decomposeSystem},
			{Role: "user", Content: prompt},
		},
	}))
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
