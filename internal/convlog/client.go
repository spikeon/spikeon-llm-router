package convlog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"
)

const ingestURL = "http://127.0.0.1:3847/conv/log"

var httpClient = &http.Client{Timeout: 2 * time.Second}

// Entry is one conversation log record.
type Entry struct {
	ID           string  `json:"id"`
	TS           string  `json:"ts"`
	Prompt       string  `json:"prompt"`
	Response     string  `json:"response"`
	ModelKey     string  `json:"model_key"`
	ModelName    string  `json:"model_name"`
	LatencyMS    float64 `json:"latency_ms"`
	Frustrated   bool    `json:"frustrated"`
	Decomposed   bool    `json:"decomposed"`
	TaskIndex    int     `json:"task_index"`
	TotalTasks   int     `json:"total_tasks"`
	FinishReason string  `json:"finish_reason"`
	HasToolCalls bool    `json:"has_tool_calls"`
}

// Log sends an entry to the lancedb-agent-memory ingest endpoint synchronously.
func Log(e Entry) {
	if e.TaskIndex == 0 && e.TotalTasks == 0 {
		e.TaskIndex = -1
		e.TotalTasks = -1
	}
	if e.FinishReason == "" {
		e.FinishReason = "stop"
	}
	e.ID = uuid.NewString()
	e.TS = time.Now().UTC().Format(time.RFC3339Nano)

	body, err := json.Marshal(e)
	if err != nil {
		return
	}
	resp, err := httpClient.Post(ingestURL, "application/json", bytes.NewReader(body))
	if err != nil {
		fmt.Printf("[conv_log] WARNING: %v\n", err)
		return
	}
	resp.Body.Close()
}

// LogAsync sends an entry in a background goroutine.
func LogAsync(e Entry) {
	go Log(e)
}
