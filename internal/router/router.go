package router

import (
	"regexp"
	"strings"
)

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
