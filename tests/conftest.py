import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def _make_completion(text: str = "mock response"):
    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.id = "mock-id"
    resp.choices = [choice]
    return resp


@pytest.fixture
def mock_ollama():
    m = MagicMock()
    m.chat.completions.create.return_value = _make_completion()
    return m


@pytest.fixture
def mock_gemini():
    return MagicMock(return_value="gemini mock response")


@pytest.fixture
def client(mock_ollama, mock_gemini):
    import api
    with patch.object(api, "ollama_client", mock_ollama), \
         patch.object(api, "gemini_chat", mock_gemini), \
         patch("api.gemini_stream", return_value=iter(["gemini mock response"])):
        yield TestClient(api.app), mock_ollama, mock_gemini
