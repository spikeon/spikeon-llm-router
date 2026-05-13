package router

import (
	"regexp"
	"strings"

	"github.com/spikeon/llm-router/internal/config"
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
	reDomain = regexp.MustCompile(`(?i)\b\w+\.(com|org|net|io|dev|ai)\b`)
	reWebRef = regexp.MustCompile(`(?i)\baccording to\b|\blook up\b|\bsearch for\b|\bfind on\b|\bon the web\b|\bonline\b`)
)

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
