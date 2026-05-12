from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import re
import time
import openai
from router import classify_prompt, is_frustrated, should_decompose, decompose_prompt
from config import MODELS, OLLAMA_BASE_URL, CAVEMAN_SYSTEM_PROMPT
from gemini_client import gemini_chat, gemini_stream

app = FastAPI()

ollama_client = openai.OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)

class Message(BaseModel):
    role: str
    content: str | list | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None

class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[Message]
    stream: bool = False
    tools: list | None = None
    tool_choice: str | dict | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | list | None = None
    response_format: dict | None = None
    seed: int | None = None

def get_last_real_prompt(messages: list) -> str | None:
    """find last user message that wasn't frustration"""
    user_msgs = [m for m in reversed(messages) if m.role == "user"]
    for msg in user_msgs[1:]:  # skip current message
        if not is_frustrated(msg.content):
            return msg.content
    return None

def _filter_think(content: str, buf: str, in_think: bool) -> tuple[str, str, bool]:
    """Strip <think>...</think> blocks from a streaming content chunk.
    buf holds partial tag chars carried between calls; in_think tracks state."""
    buf += content
    output = ""
    while buf:
        if not in_think:
            idx = buf.find("<think>")
            if idx == -1:
                safe = max(0, len(buf) - 6)
                output += buf[:safe]
                buf = buf[safe:]
                break
            output += buf[:idx]
            buf = buf[idx + 7:]
            in_think = True
        else:
            idx = buf.find("</think>")
            if idx == -1:
                buf = buf[max(0, len(buf) - 8):]
                break
            buf = buf[idx + 8:]
            in_think = False
            if buf.startswith("\n"):
                buf = buf[1:]
    return output, buf, in_think


def stream_response(model_name: str, model_key: str, messages: list, request: "ChatRequest"):
    def generate():
        response = ollama_client.chat.completions.create(
            **_ollama_kwargs(request, model_name, messages, stream=True)
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            data = {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "model": model_key,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta.content or ""},
                    "finish_reason": chunk.choices[0].finish_reason
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

def _ollama_kwargs(request: "ChatRequest", model_name: str, messages: list, stream: bool = False, num_ctx: int = None) -> dict:
    kwargs = {"model": model_name, "messages": messages, "stream": stream}
    if request.tools:
        kwargs["tools"] = request.tools
    if request.tool_choice is not None:
        kwargs["tool_choice"] = request.tool_choice
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.stop is not None:
        kwargs["stop"] = request.stop
    if request.response_format is not None:
        kwargs["response_format"] = request.response_format
    if request.seed is not None:
        kwargs["seed"] = request.seed
    if num_ctx is not None:
        kwargs["extra_body"] = {"options": {"num_ctx": num_ctx}}
    return kwargs


_VERIFY_SYSTEM = """You are a completion verifier. Compare the original request against the task responses below.
Output format (be concise):
✓ Completed: [bullet list of what was done]
✗ Missed: [anything not addressed, or "nothing"]
VERDICT: Complete / Incomplete"""


def _verify_completion(original_prompt: str, accumulated: list[tuple[str, str]]) -> str:
    parts = [f"ORIGINAL REQUEST:\n{original_prompt}\n\nTASK RESPONSES:"]
    for i, (task, resp) in enumerate(accumulated, 1):
        parts.append(f"\nTask {i} — {task}\n{resp[:600]}")
    verify_messages = [
        {"role": "system", "content": _VERIFY_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]
    try:
        resp = ollama_client.chat.completions.create(
            model=MODELS["fast"]["name"],
            messages=verify_messages,
            stream=False,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"(verify failed: {e})"


def run_decomposed(tasks: list[str], base_messages: list, request: "ChatRequest") -> dict:
    """Run decomposed tasks sequentially, passing accumulated context forward.

    Tools are forwarded to each subtask. If any subtask triggers a tool_calls
    response, it is returned immediately so the client can execute and continue.
    """
    prefix = base_messages[:-1]  # everything except the original user message
    accumulated: list[tuple[str, str]] = []  # (task, response) pairs
    wall_start = time.time()

    for i, task in enumerate(tasks):
        # Build message list: prefix + interleaved prior task/response pairs + this task
        task_messages = list(prefix)
        for prev_task, prev_resp in accumulated:
            task_messages.append({"role": "user", "content": prev_task})
            task_messages.append({"role": "assistant", "content": prev_resp})
        task_messages.append({"role": "user", "content": task})

        model_key = classify_prompt(task)
        model_display = MODELS.get(model_key, {}).get("name", model_key)
        print(f"\n[task {i + 1}/{len(tasks)}] → {model_key} ({model_display})")
        print(f"  prompt: {task[:80]}")
        t0 = time.time()

        if model_key == "gemini":
            system = next((m["content"] for m in task_messages if m["role"] == "system"), CAVEMAN_SYSTEM_PROMPT)
            history = [m for m in task_messages if m["role"] in ("user", "assistant")][:-1]
            text = gemini_chat(task, system, history)
            elapsed = time.time() - t0
            print(f"  ✓ done in {elapsed:.1f}s")
            accumulated.append((task, text))
            continue

        model_name = MODELS.get(model_key, MODELS["smart"])["name"]
        response = ollama_client.chat.completions.create(
            **_ollama_kwargs(request, model_name, task_messages)
        )
        choice = response.choices[0]
        elapsed = time.time() - t0

        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            print(f"  ↩ tool_call returned — surfacing to client")
            message = {"role": "assistant", "content": choice.message.content}
            message["tool_calls"] = [tc.model_dump() for tc in choice.message.tool_calls]
            return {
                "id": response.id,
                "object": "chat.completion",
                "model": model_key,
                "choices": [{"index": 0, "message": message, "finish_reason": choice.finish_reason}],
            }

        print(f"  ✓ done in {elapsed:.1f}s")
        accumulated.append((task, choice.message.content or ""))

    # --- Verify step ---
    original_prompt = next(
        (m["content"] for m in reversed(base_messages) if m["role"] == "user"), ""
    )
    print(f"\n[verify] comparing responses to original request...")
    t0 = time.time()
    verification = _verify_completion(original_prompt, accumulated)
    print(f"  ✓ done in {time.time() - t0:.1f}s")
    verdict_line = next((l for l in verification.splitlines() if "VERDICT" in l), "")
    print(f"  {verdict_line}")

    print(f"\n{'='*60}")
    print(f"DECOMPOSE complete — {len(accumulated)} tasks in {time.time() - wall_start:.1f}s total")
    print(f"{'='*60}\n")
    final = (accumulated[-1][1] if accumulated else "") + "\n\n---\n" + verification
    return {
        "id": "decomposed-response",
        "object": "chat.completion",
        "model": "auto-decomposed",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": final}, "finish_reason": "stop"}],
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": key, "object": "model", "owned_by": "local"}
            for key in MODELS
        ] + [
            {"id": "auto", "object": "model", "owned_by": "local"},
            {"id": "gemini", "object": "model", "owned_by": "google"},
            {"id": "worker", "object": "model", "owned_by": "local"},
        ]
    }

@app.get("/api/tags")
def ollama_tags():
    return {
        "models": [
            {"name": data["name"], "model": data["name"]}
            for data in MODELS.values()
        ]
    }

@app.get("/api/v1/models")
def ollama_v1_models():
    return list_models()

@app.get("/v1/props")
def props():
    return {}

@app.get("/props")
def props_root():
    return {}

@app.get("/version")
def version():
    return {"version": "0.1.0"}

@app.get("/v1/models/{model_id}")
def get_model(model_id: str):
    return {"id": model_id, "object": "model", "owned_by": "local"}

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    user_messages = [m for m in request.messages if m.role == "user"]
    last_prompt = user_messages[-1].content if user_messages else ""

    # orchestrator / worker: bypass consumer features, full tool-call support
    # worker also strips <think>...</think> tokens (qwen3.6 thinking mode)
    if request.model in ("orchestrator", "worker"):
        key = request.model
        model_name = MODELS[key]["name"]
        filter_think = "qwen3" in model_name.lower()

        messages = []
        for m in request.messages:
            msg = {"role": m.role, "content": m.content}
            if m.tool_call_id is not None:
                msg["tool_call_id"] = m.tool_call_id
            if m.name is not None:
                msg["name"] = m.name
            if m.tool_calls is not None:
                msg["tool_calls"] = m.tool_calls
            messages.append(msg)

        tool_names = [t.get("function", {}).get("name", "?") for t in (request.tools or [])]
        last_role = messages[-1]["role"] if messages else "?"
        last_content = str(messages[-1].get("content", ""))[:120] if messages else ""
        print(f"→ {key} ({model_name}) stream={request.stream} tools={len(tool_names)} last_role={last_role}")
        print(f"  last_msg: {last_content!r}")

        # Inject a tool directory into system prompt so the model can find key tools
        # without scanning the full list (133+ tools causes hallucination of absence)
        if tool_names and last_role == "user":
            mcp_by_prefix: dict[str, list[str]] = {}
            for t in tool_names:
                if t.startswith("mcp_"):
                    parts = t.split("_", 2)
                    prefix = parts[1] if len(parts) >= 2 else "other"
                    mcp_by_prefix.setdefault(prefix, []).append(t)
            directory_lines = ["AVAILABLE MCP TOOLS (use exact names):"]
            for prefix, tools in sorted(mcp_by_prefix.items()):
                directory_lines.append(f"  {prefix}: {', '.join(tools[:6])}{'...' if len(tools) > 6 else ''}")
            tool_directory = "\n".join(directory_lines)

            has_system = any(m.get("role") == "system" for m in messages)
            if has_system:
                for m in messages:
                    if m.get("role") == "system":
                        m["content"] = m["content"] + "\n\n" + tool_directory
                        break
            else:
                messages.insert(0, {"role": "system", "content": tool_directory})

        if request.stream:
            def _stream_agent():
                kwargs = _ollama_kwargs(request, model_name, messages, stream=True, num_ctx=131072)
                chunks_sent = 0
                finish = None
                think_buf = ""
                in_think = False
                try:
                    for chunk in ollama_client.chat.completions.create(**kwargs):
                        choice = chunk.choices[0]
                        delta = choice.delta
                        finish = choice.finish_reason
                        delta_data = {}
                        if delta.content is not None:
                            if filter_think:
                                filtered, think_buf, in_think = _filter_think(delta.content, think_buf, in_think)
                                if filtered:
                                    delta_data["content"] = filtered
                            else:
                                delta_data["content"] = delta.content
                        tc = getattr(delta, "tool_calls", None)
                        if tc:
                            delta_data["tool_calls"] = [t.model_dump() for t in tc]
                            for t in tc:
                                if t.function and t.function.name:
                                    print(f"    tool_call: {t.function.name}({str(getattr(t.function, 'arguments', ''))[:80]})")
                        if delta_data or finish:
                            data = {
                                "id": chunk.id,
                                "object": "chat.completion.chunk",
                                "model": key,
                                "choices": [{"index": 0, "delta": delta_data, "finish_reason": finish}],
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                        chunks_sent += 1
                    # flush any remaining non-think buffer
                    if filter_think and think_buf and not in_think:
                        data = {
                            "id": "flush",
                            "object": "chat.completion.chunk",
                            "model": key,
                            "choices": [{"index": 0, "delta": {"content": think_buf}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                except Exception as e:
                    print(f"  [stream error] {e}")
                print(f"  stream done: chunks={chunks_sent} finish={finish}")
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream_agent(), media_type="text/event-stream")

        # non-streaming path
        kwargs = _ollama_kwargs(request, model_name, messages, stream=False, num_ctx=131072)
        response = ollama_client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)
        content = choice.message.content or ""
        if filter_think and content:
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
        print(f"  finish={choice.finish_reason} content_len={len(content)} tool_calls={bool(tool_calls)}")
        if tool_calls:
            for tc in tool_calls:
                print(f"    tool_call: {tc.function.name}({str(tc.function.arguments)[:80]})")
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = [tc.model_dump() for tc in tool_calls]
        return {
            "id": response.id,
            "object": "chat.completion",
            "model": key,
            "choices": [{"index": 0, "message": message, "finish_reason": choice.finish_reason}],
        }

    # frustration detection
    frustrated = is_frustrated(last_prompt)
    if frustrated:
        model_key = "smart"
        last_real = get_last_real_prompt(request.messages)
        if last_real:
            print(f"→ FRUSTRATED — re-running: '{last_real[:50]}...' with smart")
            last_prompt = last_real
        else:
            print(f"→ FRUSTRATED — no prior question, using smart on current")
    else:
        model_key = classify_prompt(last_prompt) if request.model == "auto" else request.model

    model_name = MODELS.get(model_key, MODELS["smart"])["name"]

    # build messages
    has_system = any(m.role == "system" for m in request.messages)
    messages = []
    for m in request.messages:
        if m.role == "system":
            messages.append({
                "role": "system",
                "content": m.content + "\n\nADDITIONAL INSTRUCTIONS: " + CAVEMAN_SYSTEM_PROMPT
            })
        else:
            messages.append({"role": m.role, "content": m.content})

    if not has_system:
        messages.insert(0, {"role": "system", "content": CAVEMAN_SYSTEM_PROMPT})

    # when re-running last real prompt, replace the final user message in the list
    if frustrated and last_prompt != user_messages[-1].content:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i]["content"] = last_prompt
                break

    # --- Task decomposition ---
    # Only when: auto routing, not frustrated, not a tool continuation
    in_tool_continuation = request.messages and request.messages[-1].role == "tool"
    if (
        not frustrated
        and request.model == "auto"
        and not in_tool_continuation
        and should_decompose(last_prompt)
    ):
        tasks = decompose_prompt(last_prompt, ollama_client)
        if len(tasks) > 1:
            print(f"\n{'='*60}")
            print(f"DECOMPOSE → {len(tasks)} tasks")
            for j, t in enumerate(tasks, 1):
                print(f"  {j}. {t[:90]}")
            print(f"{'='*60}")
            return run_decomposed(tasks, messages, request)

    print(f"→ routing to: {model_key} ({MODELS.get(model_key, {}).get('name', model_key)})")

    # --- Gemini cloud path ---
    if model_key == "gemini":
        system = next(
            (m["content"] for m in messages if m["role"] == "system"),
            CAVEMAN_SYSTEM_PROMPT,
        )
        history = [m for m in messages if m["role"] in ("user", "assistant")]
        # drop the last user message — it's the prompt
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        if request.stream:
            def _gen_gemini():
                for text_chunk in gemini_stream(last_prompt, system, history):
                    data = {
                        "id": "gemini-stream",
                        "object": "chat.completion.chunk",
                        "model": "gemini",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": text_chunk},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_gen_gemini(), media_type="text/event-stream")

        text = gemini_chat(last_prompt, system, history)
        return {
            "id": "gemini-response",
            "object": "chat.completion",
            "model": "gemini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }

    # --- Ollama local path ---
    model_name = MODELS.get(model_key, MODELS["smart"])["name"]

    if request.stream:
        return stream_response(model_name, model_key, messages, request)

    response = ollama_client.chat.completions.create(
        **_ollama_kwargs(request, model_name, messages)
    )

    choice = response.choices[0]
    message = {"role": "assistant", "content": choice.message.content}
    if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
        message["tool_calls"] = [tc.model_dump() for tc in choice.message.tool_calls]

    return {
        "id": response.id,
        "object": "chat.completion",
        "model": model_key,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": choice.finish_reason,
        }],
    }