package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/spikeon/llm-router/config"
)

func ApiTags(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	models := make([]map[string]string, 0, len(config.Models))
	for _, def := range config.Models {
		models = append(models, map[string]string{
			"name": def.Name, "model": def.Name,
		})
	}
	json.NewEncoder(w).Encode(map[string]any{"models": models})
}
