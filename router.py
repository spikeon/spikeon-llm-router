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
    SWEAR_KEYWORDS
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
    model_name = MODELS[model_key]["name"]

    if verbose:
        print(f"→ routing to: {model_key} ({model_name})")
        print(f"→ speed: {MODELS[model_key]['tokens_per_sec']} t/s")

    # build messages
    messages = [{"role": "system", "content": system_prompt}]

    # add history if provided
    if conversation_history:
        messages.extend(conversation_history)

    # add current prompt
    messages.append({"role": "user", "content": prompt})

    # call ollama
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    text = response.choices[0].message.content
    return text, model_key

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