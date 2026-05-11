"""
Tests for the plan-and-execute decomposition behavior.

Scenario: user sends a single message that contains multiple distinct
requests. The system should recognize this, break it into steps,
run each through the appropriate model, and verify completion at the end.
"""
import pytest
from unittest.mock import patch, MagicMock
from router import should_decompose


# --- Unit: recognizing multi-step prompts ---

def test_single_question_is_not_decomposed():
    assert should_decompose("what does a left join do in SQL") is False


def test_short_prompt_is_not_decomposed():
    assert should_decompose("hello") is False


def test_prompt_with_multiple_requests_is_decomposed():
    assert should_decompose(
        "look at my bills and tell me what I owe, "
        "then write a script to parse that as CSV, "
        "then explain how to read a budget sheet"
    ) is True


def test_prompt_with_sequential_steps_is_decomposed():
    assert should_decompose(
        "find the bug in my code and then fix it and also add a test for it"
    ) is True


def test_long_single_topic_question_is_not_decomposed():
    # Long doesn't mean multi-step — this is one cohesive question
    assert should_decompose(
        "can you give me a detailed explanation of how the event loop works "
        "in Node.js and why blocking the thread is a problem"
    ) is False


# --- API behavior: decomposition pipeline ---

def test_multi_step_prompt_calls_multiple_models(client):
    """A prompt with several distinct requests should result in more than one
    model being called, each handling its own subtask."""
    c, mock_ollama, _ = client

    with patch("api.decompose_prompt") as mock_decompose, \
         patch("api.should_decompose", return_value=True):
        mock_decompose.return_value = [
            "write a Python function that reverses a string",
            "explain what time complexity means",
        ]
        response = c.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "write a Python function that reverses a string and then explain what time complexity means"}],
        })

    assert response.status_code == 200
    assert mock_ollama.chat.completions.create.call_count >= 2


def test_tool_call_mid_decomposition_is_returned_immediately(client):
    """If a subtask triggers a tool call, the decomposition loop should stop
    and hand the tool call back to the client — not try to continue."""
    c, mock_ollama, _ = client

    tool_call_mock = MagicMock()
    tool_call_mock.id = "call_abc"
    tool_call_mock.model_dump.return_value = {"id": "call_abc", "type": "function"}

    tool_response = MagicMock()
    tool_response.id = "resp-tool"
    tool_response.choices[0].message.content = None
    tool_response.choices[0].message.tool_calls = [tool_call_mock]
    tool_response.choices[0].finish_reason = "tool_calls"
    mock_ollama.chat.completions.create.return_value = tool_response

    with patch("api.decompose_prompt") as mock_decompose, \
         patch("api.should_decompose", return_value=True):
        mock_decompose.return_value = [
            "look up the user's account balance",
            "send them a summary email",  # should never run
        ]
        response = c.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "look up my balance and then email me a summary"}],
            "tools": [{"type": "function", "function": {"name": "get_balance", "parameters": {}}}],
        })

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"].get("tool_calls") is not None
    # Only one model call — stopped at the tool call, didn't continue to task 2
    assert mock_ollama.chat.completions.create.call_count == 1


def test_tool_continuation_skips_decomposition(client):
    """When the client sends back a tool result, that follow-up should go
    straight to the model — not re-trigger the decomposition planner."""
    c, mock_ollama, _ = client

    # Use a multi-step prompt that would normally trigger decomposition,
    # but this time it arrives as a tool continuation — should be a single call
    response = c.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [
            {"role": "user", "content": "debug my Python function and then write a test for it"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "def add(a, b): return a - b"},
        ],
    })

    assert response.status_code == 200
    # One direct model call — decomposition was not triggered
    assert mock_ollama.chat.completions.create.call_count == 1


def test_decomposition_appends_completion_check(client):
    """After all subtasks finish, the response should include a verification
    section confirming what was and wasn't addressed from the original request."""
    c, mock_ollama, _ = client

    with patch("api.decompose_prompt") as mock_decompose, \
         patch("api.should_decompose", return_value=True):
        mock_decompose.return_value = [
            "write a Python function that reverses a string",
            "explain what time complexity means",
        ]
        response = c.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "write a Python function that reverses a string and then explain what time complexity means"}],
        })

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    assert "---" in content  # separator before verification section
