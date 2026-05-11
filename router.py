import openai
from config import (
    MODELS,
    OLLAMA_BASE_URL,
    CAVEMAN_SYSTEM_PROMPT,
    TOKEN_THRESHOLD_SNAPPY,
    TOKEN_THRESHOLD_LONG,
    CODE_KEYWORDS,
    REASONING_KEYWORDS,
    BALANCED_KEYWORDS,
    MEMORY_KEYWORDS,
    SWEAR_KEYWORDS,
    FINANCE_KEYWORDS,
    GOOGLE_KEYWORDS,
)
import re

client = openai.OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)


def count_tokens(text: str) -> float:
    return len(text.split()) * 1.3


def classify_prompt(prompt: str) -> str:
    p = prompt.lower()
    tokens = count_tokens(prompt)

    # finance or Google Workspace → Gemini cloud
    if any(w in p for w in FINANCE_KEYWORDS) or any(w in p for w in GOOGLE_KEYWORDS):
        return "gemini"

    # pattern: setting or retrieving personal info → needs tool-capable model
    if re.search(
        r"\b(my|your) .+ is\b"           # "my quest is..."
        r"|\bi (am|prefer|like|hate|love|use|have)\b"  # "I am / I prefer..."
        r"|\bwhat(\'?s| is| are) (my|your)\b"          # "what is my X?"
        r"|\bdo you (know|remember)\b"                  # "do you remember...?"
        r"|\bwhat did i\b"                              # "what did I say/tell you"
        r"|\btell me (about )?(my|your)\b",             # "tell me my X"
        p
    ):
        return "memory-fast"

    # memory/identity keywords
    if any(w in p for w in MEMORY_KEYWORDS):
        return "memory-fast"

    # web lookup → needs tool-capable model
    if re.search(r"\b\w+\.(com|org|net|io|dev|ai)\b", p):  # URL/domain mentioned
        return "balanced"
    if re.search(r"\baccording to\b|\blook up\b|\bsearch for\b|\bfind on\b|\bon the web\b|\bonline\b", p):
        return "balanced"

    # code → mistral
    if any(w in p for w in CODE_KEYWORDS):
        return "coder"

    # long → smart
    if tokens > TOKEN_THRESHOLD_LONG:
        return "smart"

    # reasoning → thinker
    if any(w in p for w in REASONING_KEYWORDS):
        return "thinker"

    # write/explain → balanced
    if any(w in p for w in BALANCED_KEYWORDS):
        return "balanced"

    # short → fast
    if tokens < TOKEN_THRESHOLD_SNAPPY:
        return "fast"

    return "fast"


def chat(
    prompt: str,
    override_model: str = None,
    system_prompt: str = CAVEMAN_SYSTEM_PROMPT,
    conversation_history: list = None,
    verbose: bool = False
) -> tuple[str, str]:
    """
    Returns (response_text, model_used)
    """

    # pick model
    model_key = override_model if override_model else classify_prompt(prompt)

    if verbose:
        print(f"→ routing to: {model_key} ({MODELS[model_key]['name']})")
        print(f"→ speed: {MODELS[model_key]['tokens_per_sec']} t/s")

    # Gemini cloud path
    if model_key == "gemini":
        from gemini_client import gemini_chat
        text = gemini_chat(prompt, system_prompt, conversation_history or [])
        return text, "gemini"

    model_name = MODELS[model_key]["name"]

    # build messages
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    text = response.choices[0].message.content
    return text, model_key

_DECOMPOSE_CONNECTORS = [
    "and then", "and also", "and make", "and change", "and move",
    "and set", "and update", "and add", "and remove", "and fix",
    "as well as", "additionally", "furthermore", "after that",
    ", and ", " then ", " also ",
]

_DECOMPOSE_SYSTEM = """Break the user's request into a numbered list of atomic, sequential tasks.
Rules:
- One clear action per task
- Logical order of operations (discover before modifying)
- Use $variables for values found in earlier steps
- Output ONLY the numbered list, nothing else"""

DECOMPOSE_TOKEN_MIN = 20


def should_decompose(prompt: str) -> bool:
    if count_tokens(prompt) < DECOMPOSE_TOKEN_MIN:
        return False
    p = prompt.lower()
    connector_hits = sum(1 for c in _DECOMPOSE_CONNECTORS if c in p)
    sentences = [s.strip() for s in re.split(r"[.!?]", prompt) if len(s.strip()) > 5]
    return connector_hits >= 2 or (connector_hits >= 1 and len(sentences) >= 2) or len(sentences) >= 3


def decompose_prompt(prompt: str, ollama_client) -> list[str]:
    try:
        response = ollama_client.chat.completions.create(
            model=MODELS["snappy"]["name"],
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        text = response.choices[0].message.content or ""
        tasks = []
        for line in text.split("\n"):
            m = re.match(r"^\d+[\.\)]\s+(.+)", line.strip())
            if m:
                tasks.append(m.group(1).strip())
        return tasks if len(tasks) > 1 else [prompt]
    except Exception:
        return [prompt]


def is_frustrated(prompt: str) -> bool:
    p = prompt.lower()
    
    # check swearing
    if any(w in p for w in SWEAR_KEYWORDS):
        return True
    
    # check yelling — >50% caps alpha chars
    alpha = [c for c in prompt if c.isalpha()]
    if len(alpha) > 3 and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.5:
        return True
    
    return False