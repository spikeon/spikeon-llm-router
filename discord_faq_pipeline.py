#!/usr/bin/env python3
"""
Discord → FAQ gap analyzer.

Fetches recent messages from ExoGuitar Discord channels via REST API,
compares against existing FAQ files, and outputs FAQ candidates.

Results: /tmp/discord_faq_candidates.md
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
if not BOT_TOKEN:
    sys.exit("DISCORD_BOT_TOKEN env var not set")

CHANNELS = {
    "ExoGuitar":              "1304105121768149113",
    "ExoBass":                "1309259841432195103",
    "Exo-Builds":             "1379072332139593798",
    "General":                "1304105003098832951",
    "Welcome-and-Announcements": "1304105003098832948",
}

MESSAGES_PER_CHANNEL = 100

FAQ_FILES = [
    Path("/home/spikeon/Dev/ExoWorks/exoguitar/FAQ.md"),
    Path("/home/spikeon/Dev/ExoWorks/exobass/FAQ.md"),
]

ROUTER_URL = "http://localhost:11435/v1/chat/completions"
MODEL      = "smart"  # gemma4:31b — best quality for analysis

RAW_OUT  = Path("/tmp/discord_faq_raw.md")
FINAL_OUT = Path("/tmp/discord_faq_candidates.md")


# ── discord fetch ─────────────────────────────────────────────────────────────

def fetch_messages(channel_id: str, limit: int = 100) -> list[dict]:
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit={limit}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bot {BOT_TOKEN}",
            "User-Agent": "ExoFAQBot/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  HTTP {e.code} on channel {channel_id}: {body[:200]}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  Error fetching {channel_id}: {e}", file=sys.stderr)
        return []


def format_message(msg: dict) -> str | None:
    # Skip bots and system messages
    if msg.get("type", 0) != 0:
        return None
    author = msg.get("author", {})
    if author.get("bot", False):
        return None
    content = msg.get("content", "").strip()
    if not content:
        return None
    ts = msg.get("timestamp", "")[:16]  # trim sub-seconds
    username = author.get("username", "unknown")
    return f"- [{ts}] @{username}: {content}"


# ── llm call ──────────────────────────────────────────────────────────────────

def call_model(system: str, user: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": 4096,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        ROUTER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[model call failed: {e}]"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Discord FAQ pipeline")

    # Step 1 — fetch messages
    raw_sections = []
    total_msgs = 0

    for name, cid in CHANNELS.items():
        print(f"  Fetching #{name} ({cid}) ...", end=" ", flush=True)
        msgs = fetch_messages(cid, MESSAGES_PER_CHANNEL)
        # msgs arrive newest-first; reverse for chronological
        msgs = list(reversed(msgs))
        lines = [fmt for m in msgs if (fmt := format_message(m))]
        count = len(lines)
        total_msgs += count
        print(f"{count} messages")

        section = f"## Channel: {name} ({cid})\n"
        section += "\n".join(lines) if lines else "(no messages)"
        raw_sections.append(section)

    raw_text = "\n\n".join(raw_sections)
    RAW_OUT.write_text(
        f"# Discord Raw Messages\nFetched: {datetime.now().isoformat()}\n"
        f"Total messages: {total_msgs}\n\n---\n\n{raw_text}"
    )
    print(f"  Raw data written → {RAW_OUT}  ({total_msgs} total messages)")

    # Step 2 — read existing FAQs
    faq_content = ""
    for faq_path in FAQ_FILES:
        if faq_path.exists():
            faq_content += f"\n\n### {faq_path.name} ({faq_path.parent.name})\n\n"
            faq_content += faq_path.read_text()
    if not faq_content:
        faq_content = "(no existing FAQ files found)"

    # Step 3 — analyze
    print(f"  Calling model ({MODEL}) for FAQ gap analysis ...")

    system_prompt = """You are an expert at analyzing community Discord conversations to identify FAQ candidates.

Your job:
1. Read the Discord messages provided
2. Compare against the EXISTING FAQ content
3. Identify what's missing, outdated, or needs expansion
4. Produce actionable FAQ candidates

Rules:
- Focus on questions asked by multiple users OR questions that got long detailed answers
- Look for confusion patterns: "how do I", "where is", "does it", "can I", "I can't figure out"
- Recurring topics: strings, tuning, pickups, builds, compatibility, ordering, shipping, setup, filament, printing
- Ignore bot messages, spam, casual chat with no informational value
- Don't duplicate what's already well-covered in existing FAQs
- Be specific — synthesize real answers from what was discussed, don't be vague"""

    user_prompt = f"""## EXISTING FAQ CONTENT
{faq_content}

---

## DISCORD MESSAGES ({total_msgs} total from 5 channels)
{raw_text}

---

## YOUR TASK

Analyze the Discord messages and identify FAQ gaps. Produce output in this exact format:

# ExoGuitar Discord FAQ Candidates
Generated: {datetime.now().strftime('%Y-%m-%d')}
Channels scanned: {len(CHANNELS)}
Messages analyzed: {total_msgs}

---

## HIGH CONFIDENCE
(Questions/topics seen repeatedly or that got detailed answers; clearly missing from or only vaguely covered in existing FAQ)

### Q: [question as naturally asked]
**A:** [draft answer synthesized from discussion]
**Source:** #channel-name
**Seen:** X times

---

## MEDIUM CONFIDENCE
(Mentioned a few times, or touched on tangentially in existing FAQ but could be expanded)

[same format]

---

## LOW CONFIDENCE
(Mentioned once, or borderline FAQ-worthy)

[same format]

---

## EXISTING FAQ GAPS/UPDATES
List any existing FAQ answers that seem outdated, incomplete, or contradicted by Discord discussion.

If no significant FAQ gaps exist, say so clearly."""

    result = call_model(system_prompt, user_prompt)

    FINAL_OUT.write_text(result)
    print(f"  FAQ candidates written → {FINAL_OUT}")

    # Step 4 — summary
    lines_in_result = result.count("\n")
    print(f"\n{'='*60}")
    print(f"Done. {total_msgs} messages from {len(CHANNELS)} channels analyzed.")
    print(f"Raw data:  {RAW_OUT}")
    print(f"FAQ output: {FINAL_OUT}")
    print(f"\n--- PREVIEW (first 80 lines) ---")
    preview_lines = result.split("\n")[:80]
    print("\n".join(preview_lines))
    if len(result.split("\n")) > 80:
        print(f"\n... ({lines_in_result - 80} more lines in {FINAL_OUT})")


if __name__ == "__main__":
    main()
