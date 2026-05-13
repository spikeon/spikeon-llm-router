package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/spikeon/llm-router/config"
)

func ListModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	data := make([]map[string]any, 0, len(config.Models)+3)
	for key := range config.Models {
		data = append(data, map[string]any{
			"id": key, "object": "model", "owned_by": "local",
		})
	}
	data = append(data,
		map[string]any{"id": "auto", "object": "model", "owned_by": "local"},
		map[string]any{"id": "gemini", "object": "model", "owned_by": "google"},
		map[string]any{"id": "worker", "object": "model", "owned_by": "local"},
	)
	json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
}

func GetModel(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": r.PathValue("model_id"), "object": "model", "owned_by": "local",
	})
}
