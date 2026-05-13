package router

import (
	"strings"
	"unicode"

	"github.com/spikeon/llm-router/internal/config"
)

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
