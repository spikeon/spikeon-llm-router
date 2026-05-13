package config

type ModelDef struct {
	Name         string
	TokensPerSec int
	Description  string
}

var Models = map[string]ModelDef{
	"snappy":       {"qwen2.5:0.5b", 397, "Trivial/factual/short"},
	"fast":         {"gemma4:e2b", 145, "General chat"},
	"memory-fast":  {"gemma4:e2b", 145, "Memory/preference saving (tool-capable)"},
	"coder":        {"mistral:7b", 111, "Code/debug/math"},
	"balanced":     {"gemma4:e4b", 105, "Write/summarize/explain"},
	"thinker":      {"qwen3.6:latest", 21, "Deep reasoning/analysis"},
	"smart":        {"gemma4:31b", 9, "Hard/long/complex"},
	"orchestrator": {"qwen3.6:latest", 21, "Hermes main agent: strong reasoning + tool selection, think-tokens filtered"},
	"worker":       {"qwen3.6:latest", 21, "Heavy tool-calling subagent: reasoning + MCP tool use, think-tokens filtered"},
	"gemini":       {"gemini-2.0-flash", 150, "Google cloud: Drive/Sheets/Gmail + finance (Bills sheet)"},
}

const (
	OllamaBaseURL        = "http://localhost:11434/v1"
	TokenThresholdSnappy = 15
	TokenThresholdLong   = 600
	BillsSheetName       = "Bills"
)

const CavemanSystemPrompt = `Ultra-compressed response rules:
- Drop all articles (a, an, the)
- Drop filler words (very, really, just, simply)
- No pleasantries or preamble
- Short punchy fragments
- Max signal, zero fluff
- Use symbols where possible (→, +, =, &)
- Lists over paragraphs always
- Never restate question
- Jump straight to answer`

var CodeKeywords = []string{
	"code", "debug", "function", "class", "error",
	"fix", "script", "python", "javascript", "bash",
	"sql", "programming", "implement", "refactor",
}

var ReasoningKeywords = []string{
	"analyze", "reason", "compare", "pros", "cons",
	"plan", "strategy", "think through", "evaluate",
	"should i", "which is better", "tradeoffs",
}

var BalancedKeywords = []string{
	"summarize", "rewrite", "translate", "write",
	"explain", "essay", "draft", "describe",
	"reference", "media", "movie", "book", "quote", "joke",
}

var MemoryKeywords = []string{
	"remember", "your name", "my name", "call you",
	"forget", "recall", "who are you", "who am i",
}

var SwearKeywords = []string{
	"fuck", "shit", "damn", "hell", "ass", "crap", "wtf", "bullshit",
}

var FinanceKeywords = []string{
	"bill", "bills", "money", "payment", "budget", "expense", "expenses",
	"income", "spend", "spent", "cost", "costs", "rent", "salary",
	"debt", "loan", "bank", "account", "savings", "invest", "investing",
	"stock", "finance", "financial", "dollar", "pay", "paid", "owe",
	"credit card", "debit", "insurance", "subscription", "overdue",
	"due date", "balance", "wallet", "refund", "charge", "afford",
	"transaction", "monthly", "annual fee",
}

var GoogleKeywords = []string{
	"google drive", "my drive", "google sheet", "google sheets",
	"spreadsheet", "gmail", "my email", "check my email", "inbox",
	"sent mail", "google doc", "google docs", "my files", "drive file",
}

var EmailTerms = []string{"email", "gmail", "inbox", "mail", "sent mail"}
