package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

const maxAPIErrorMessageRunes = 4096

// newUpstreamHTTPError builds a client-facing error from a failed chat/completions HTTP response.
// streamCtx is prefixed when non-empty (e.g. "stream").
func newUpstreamHTTPError(streamCtx string, status int, body []byte) error {
	var core string
	if msg := extractAPIErrorMessage(body); msg != "" {
		core = formatWithStatus(status, msg)
	} else {
		core = fmt.Sprintf("HTTP %d: %s", status, truncateBody(body, 900))
	}
	if streamCtx != "" {
		return fmt.Errorf("%s: %s", streamCtx, core)
	}
	return fmt.Errorf("%s", core)
}

func formatWithStatus(status int, msg string) string {
	switch status {
	case http.StatusTooManyRequests:
		return fmt.Sprintf("rate limit or quota exceeded (HTTP %d): %s", status, msg)
	case http.StatusUnauthorized:
		return fmt.Sprintf("authentication failed (HTTP %d): %s", status, msg)
	case http.StatusForbidden:
		return fmt.Sprintf("forbidden (HTTP %d): %s", status, msg)
	case http.StatusBadRequest:
		return fmt.Sprintf("bad request (HTTP %d): %s", status, msg)
	default:
		if status >= 500 {
			return fmt.Sprintf("upstream server error (HTTP %d): %s", status, msg)
		}
		return fmt.Sprintf("upstream error (HTTP %d): %s", status, msg)
	}
}

// extractAPIErrorMessage pulls a human-readable message from Google/OpenAI-style JSON error bodies.
// Gemini often returns: [{"error":{"code":429,"message":"..."}}]
func extractAPIErrorMessage(body []byte) string {
	body = bytes.TrimSpace(body)
	if len(body) == 0 {
		return ""
	}

	type errObj struct {
		Message string `json:"message"`
	}
	type envelope struct {
		Error *errObj `json:"error"`
	}

	var asBatch []envelope
	if err := json.Unmarshal(body, &asBatch); err == nil {
		for _, item := range asBatch {
			if item.Error != nil {
				if m := strings.TrimSpace(item.Error.Message); m != "" {
					return limitErrorMessage(compactWS(m))
				}
			}
		}
	}

	var one envelope
	if err := json.Unmarshal(body, &one); err == nil && one.Error != nil {
		if m := strings.TrimSpace(one.Error.Message); m != "" {
			return limitErrorMessage(compactWS(m))
		}
	}

	return ""
}

func compactWS(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func limitErrorMessage(s string) string {
	if len(s) <= maxAPIErrorMessageRunes {
		return s
	}
	r := []rune(s)
	if len(r) <= maxAPIErrorMessageRunes {
		return s
	}
	return string(r[:maxAPIErrorMessageRunes]) + "…"
}
