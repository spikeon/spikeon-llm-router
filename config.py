# Model definitions with real benchmarked speeds
MODELS = {
    "snappy": {
        "name": "qwen2.5:0.5b",
        "tokens_per_sec": 397,
        "description": "Trivial/factual/short"
    },
    "fast": {
        "name": "gemma4:e2b",
        "tokens_per_sec": 145,
        "description": "General chat"
    },
    "memory-fast": {
        "name": "gemma4:e2b",
        "tokens_per_sec": 145,
        "description": "Memory/preference saving (tool-capable)"
    },
    "coder": {
        "name": "mistral:7b",
        "tokens_per_sec": 111,
        "description": "Code/debug/math"
    },
    "balanced": {
        "name": "gemma4:e4b",
        "tokens_per_sec": 105,
        "description": "Write/summarize/explain"
    },
    "thinker": {
        "name": "qwen3.6:latest",
        "tokens_per_sec": 21,
        "description": "Deep reasoning/analysis"
    },
    "smart": {
        "name": "gemma4:31b",
        "tokens_per_sec": 9,
        "description": "Hard/long/complex"
    },
}

# Ollama endpoint
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Caveman ultra system prompt
CAVEMAN_SYSTEM_PROMPT = """Ultra-compressed response rules:
- Drop all articles (a, an, the)
- Drop filler words (very, really, just, simply)
- No pleasantries or preamble
- Short punchy fragments
- Max signal, zero fluff
- Use symbols where possible (→, +, =, &)
- Lists over paragraphs always
- Never restate question
- Jump straight to answer"""

# Routing thresholds
TOKEN_THRESHOLD_SNAPPY = 15    # words
TOKEN_THRESHOLD_LONG = 600     # words → smart model

# Keywords
CODE_KEYWORDS = [
    "code", "debug", "function", "class", "error",
    "fix", "script", "python", "javascript", "bash",
    "sql", "programming", "implement", "refactor"
]

REASONING_KEYWORDS = [
    "analyze", "reason", "compare", "pros", "cons",
    "plan", "strategy", "think through", "evaluate",
    "should i", "which is better", "tradeoffs"
]

BALANCED_KEYWORDS = [
    "summarize", "rewrite", "translate", "write",
    "explain", "essay", "draft", "describe",
    "reference", "media", "movie", "book", "quote", "joke"
]

MEMORY_KEYWORDS = [
    "remember", "your name", "my name", "call you",
    "forget", "recall", "who are you", "who am i"
]

SWEAR_KEYWORDS = [
    "fuck", "shit", "damn", "hell", "ass", "crap", "wtf", "bullshit"
]

# Google Gemini cloud model — handles Drive, Sheets, Gmail, and finance
MODELS["gemini"] = {
    "name": "gemini-2.0-flash",
    "tokens_per_sec": 150,
    "description": "Google cloud: Drive/Sheets/Gmail + finance (Bills sheet)",
}

BILLS_SHEET_NAME = "Bills"

FINANCE_KEYWORDS = [
    "bill", "bills", "money", "payment", "budget", "expense", "expenses",
    "income", "spend", "spent", "cost", "costs", "rent", "salary",
    "debt", "loan", "bank", "account", "savings", "invest", "investing",
    "stock", "finance", "financial", "dollar", "pay", "paid", "owe",
    "credit card", "debit", "insurance", "subscription", "overdue",
    "due date", "balance", "wallet", "refund", "charge", "afford",
    "transaction", "monthly", "annual fee",
]

GOOGLE_KEYWORDS = [
    "google drive", "my drive", "google sheet", "google sheets",
    "spreadsheet", "gmail", "my email", "check my email", "inbox",
    "sent mail", "google doc", "google docs", "my files", "drive file",
]
