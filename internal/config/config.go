package config

import (
	"os"
	"strings"
)

type ModelDef struct {
	Name         string
	Provider     string // key into Providers; empty => DefaultProvider ("local")
	APIKeyEnv    string // optional: env var whose value is sent as Bearer token (OpenAI-compat providers)
	TokensPerSec int
	Description  string
}

// DefaultProvider is the Providers map key used when ModelDef.Provider is empty.
const DefaultProvider = "local"

const (
	OllamaBaseURL        = "http://localhost:11434/v1"
	TokenThresholdSnappy = 15
	TokenThresholdLong   = 600
	BillsSheetName       = "Bills"
)

// DefaultParserName selects the built-in OpenAI-compatible completion normalizer when ProviderDef.Parser is empty.
const DefaultParserName = "openai_compat"

// ProviderDef configures an upstream OpenAI-compatible endpoint. Parser names a registered CompletionParser
// (see internal/providers/ollama); empty Parser defaults to DefaultParserName.
type ProviderDef struct {
	BaseURL string
	Parser  string
}

// Providers maps logical provider keys to endpoint + parser metadata.
//
// Override only the "local" entry BaseURL at runtime with OLLAMA_BASE_URL (existing Docker / dev behavior).
//
// "google_openai" targets Gemini via the Google AI OpenAI compatibility layer; set APIKeyEnv to GEMINI_API_KEY on models that use it.
var Providers = map[string]ProviderDef{
	"local": {BaseURL: OllamaBaseURL},
	// https://ai.google.dev/gemini-api/docs/openai
	"google_openai": {BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai", Parser: "google_openai"},
}

var Models = map[string]ModelDef{
	"snappy":       {Name: "qwen2.5:0.5b", TokensPerSec: 397, Description: "Trivial/factual/short"},
	"fast":         {Name: "gemma4:e2b", TokensPerSec: 145, Description: "General chat"},
	"memory-fast":  {Name: "gemma4:e2b", TokensPerSec: 145, Description: "Memory/preference saving (tool-capable)"},
	"coder":        {Name: "mistral:7b", TokensPerSec: 111, Description: "Code/debug/math"},
	"balanced":     {Name: "gemma4:e4b", TokensPerSec: 105, Description: "Write/summarize/explain"},
	"thinker":      {Name: "qwen3.6:latest", TokensPerSec: 21, Description: "Deep reasoning/analysis"},
	"smart":        {Name: "gemma4:31b", TokensPerSec: 9, Description: "Hard/long/complex"},
	"orchestrator": {Name: "gemini-3-flash-preview", Provider: "google_openai", APIKeyEnv: "GEMINI_API_KEY", TokensPerSec: 150, Description: "Hermes main agent: strong reasoning + tool selection (Gemini OpenAI-compat); think-tokens filtered if model emits them"},
	"worker":       {Name: "qwen3.6:latest", TokensPerSec: 21, Description: "Heavy tool-calling subagent: reasoning + MCP tool use, think-tokens filtered"},
	"gemini":       {Name: "gemini-3-flash-preview", TokensPerSec: 150, Description: "Google cloud: Drive/Sheets/Gmail + finance (Bills sheet)"},
}

// ProviderKey returns the config key for looking up a base URL.
func (m ModelDef) ProviderKey() string {
	if m.Provider != "" {
		return m.Provider
	}
	return DefaultProvider
}

// LookupProvider returns merged provider config for a key, falling back to DefaultProvider then a minimal local default.
func LookupProvider(providerKey string) ProviderDef {
	if providerKey == "" {
		providerKey = DefaultProvider
	}
	if def, ok := Providers[providerKey]; ok && strings.TrimSpace(def.BaseURL) != "" {
		return def
	}
	if providerKey != DefaultProvider {
		return LookupProvider(DefaultProvider)
	}
	return ProviderDef{BaseURL: OllamaBaseURL}
}

// ParserKeyForProvider returns the CompletionParser registration name for this provider (from ProviderDef.Parser or DefaultParserName).
func ParserKeyForProvider(providerKey string) string {
	def := LookupProvider(providerKey)
	if strings.TrimSpace(def.Parser) != "" {
		return strings.TrimSpace(def.Parser)
	}
	return DefaultParserName
}

// ProviderBaseURL resolves the OpenAI-compatible API root for a provider key.
// Unknown keys fall back to the default provider chain.
// OLLAMA_BASE_URL overrides only when resolving DefaultProvider ("local").
func ProviderBaseURL(providerKey string) string {
	if providerKey == "" {
		providerKey = DefaultProvider
	}
	if providerKey == DefaultProvider {
		if u := os.Getenv("OLLAMA_BASE_URL"); strings.TrimSpace(u) != "" {
			return normalizeAPIBase(u)
		}
	}
	def := LookupProvider(providerKey)
	if strings.TrimSpace(def.BaseURL) != "" {
		return normalizeAPIBase(def.BaseURL)
	}
	if providerKey != DefaultProvider {
		return ProviderBaseURL(DefaultProvider)
	}
	return normalizeAPIBase(OllamaBaseURL)
}

func normalizeAPIBase(u string) string {
	return strings.TrimRight(strings.TrimSpace(u), "/")
}
