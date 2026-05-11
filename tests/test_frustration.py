"""
Tests for frustration detection and escalation behavior.

Scenario: user asks something, gets a bad answer, starts getting angry.
The system should recognize the frustration and retry with a smarter model.
"""
import pytest
from router import is_frustrated
from config import MODELS


# --- Unit: detecting when a user is frustrated ---

def test_all_caps_message_is_frustrated():
    assert is_frustrated("WHY IS THIS NOT WORKING") is True


def test_mixed_angry_caps_is_frustrated():
    # More than half the letters are uppercase
    assert is_frustrated("THAT IS COMPLETELY WRONG omg") is True


def test_swearing_is_frustrated():
    assert is_frustrated("wtf is this supposed to do") is True


def test_calm_question_is_not_frustrated():
    assert is_frustrated("why isn't this working as expected") is False


def test_normal_question_is_not_frustrated():
    assert is_frustrated("can you explain how promises work in JavaScript") is False


def test_single_caps_word_is_not_frustrated():
    # An acronym or emphasis word doesn't count as yelling
    assert is_frustrated("what does API stand for") is False


# --- API behavior: escalation to smarter model ---

def test_frustrated_user_gets_smarter_model(client):
    """When the user's follow-up shows they're angry, re-run their original
    question with the most capable model instead of responding to the anger."""
    c, mock_ollama, _ = client
    response = c.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [
            {"role": "user", "content": "what's the best way to structure a Python project"},
            {"role": "assistant", "content": "just put everything in one file"},
            {"role": "user", "content": "THAT MAKES NO SENSE AT ALL WHY WOULD YOU SAY THAT"},
        ],
    })
    assert response.status_code == 200
    call_kwargs = mock_ollama.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == MODELS["smart"]["name"]


def test_frustrated_user_reruns_original_question_not_the_anger(client):
    """The angry message itself should not be what gets sent to the smarter model —
    the original unanswered question should be retried."""
    c, mock_ollama, _ = client
    c.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [
            {"role": "user", "content": "how do database indexes work"},
            {"role": "assistant", "content": "they are like bookmarks"},
            {"role": "user", "content": "THAT IS A TERRIBLE EXPLANATION"},
        ],
    })
    call_kwargs = mock_ollama.chat.completions.create.call_args[1]
    last_user = next(m for m in reversed(call_kwargs["messages"]) if m["role"] == "user")
    assert "index" in last_user["content"].lower()
    assert "terrible" not in last_user["content"].lower()
