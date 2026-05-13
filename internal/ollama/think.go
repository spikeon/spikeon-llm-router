package ollama

import "strings"

// ThinkFilter strips <think>…</think> blocks from a streaming response.
type ThinkFilter struct {
	buf     string
	inThink bool
}

// Write processes a streaming chunk and returns the filtered output.
func (f *ThinkFilter) Write(chunk string) string {
	f.buf += chunk
	out := strings.Builder{}
	for {
		if !f.inThink {
			idx := strings.Index(f.buf, "<think>")
			if idx == -1 {
				safe := len(f.buf) - 6
				if safe < 0 {
					safe = 0
				}
				out.WriteString(f.buf[:safe])
				f.buf = f.buf[safe:]
				break
			}
			out.WriteString(f.buf[:idx])
			f.buf = f.buf[idx+7:]
			f.inThink = true
		} else {
			idx := strings.Index(f.buf, "</think>")
			if idx == -1 {
				keep := len(f.buf) - 8
				if keep < 0 {
					keep = 0
				}
				f.buf = f.buf[keep:]
				break
			}
			f.buf = f.buf[idx+8:]
			f.inThink = false
			if strings.HasPrefix(f.buf, "\n") {
				f.buf = f.buf[1:]
			}
		}
	}
	return out.String()
}

// Flush returns any buffered non-think content and resets state.
func (f *ThinkFilter) Flush() string {
	if !f.inThink {
		out := f.buf
		f.buf = ""
		return out
	}
	return ""
}

// StripThink removes all <think>…</think> blocks from a complete string.
func StripThink(s string) string {
	f := &ThinkFilter{}
	out := f.Write(s)
	out += f.Flush()
	return strings.TrimSpace(out)
}
