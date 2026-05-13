package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"

	"github.com/spikeon/llm-router/internal/handlers"
)

func main() {
	if wd, err := os.Getwd(); err == nil {
		loadDotEnv(filepath.Join(wd, ".env"))
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "11435"
	}

	mux := http.NewServeMux()

	mux.HandleFunc("GET /version", handlers.Version)
	mux.HandleFunc("GET /v1/props", handlers.Props)
	mux.HandleFunc("GET /props", handlers.Props)
	mux.HandleFunc("GET /api/tags", handlers.ApiTags)
	mux.HandleFunc("GET /v1/models", handlers.ListModels)
	mux.HandleFunc("GET /api/v1/models", handlers.ListModels)
	mux.HandleFunc("GET /v1/models/{model_id}", handlers.GetModel)
	mux.HandleFunc("POST /v1/chat/completions", handlers.Chat)

	fmt.Printf("LLM Router listening on :%s\n", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		fmt.Fprintf(os.Stderr, "server error: %v\n", err)
		os.Exit(1)
	}
}
