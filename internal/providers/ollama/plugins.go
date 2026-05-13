package ollama

// Custom CompletionParser registration
//
// In-repo: add parse_<name>.go in this package (same as parse_google_openai.go): define a type that
// implements CompletionParser, then RegisterCompletionParser("name", ...) from init(). Set
// config.ProviderDef.Parser to "name" for providers that need it.
//
// Out-of-repo Go code: import this package, implement CompletionParser, call RegisterCompletionParser
// from init(), and blank-import that package from cmd/llm-router so init runs.
