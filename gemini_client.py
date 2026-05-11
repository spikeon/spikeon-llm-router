import json
import os
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from config import BILLS_SHEET_NAME, FINANCE_KEYWORDS

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
]

# Hermes stores Google OAuth tokens here when the workspace skill is set up
_HERMES_TOKEN_PATH = os.path.expanduser("~/.hermes/google_token.json")
_HERMES_CREDS_PATH = os.path.expanduser("~/.hermes/google_client_secret.json")
# Fallback locations (standalone router setup)
_TOKEN_PATH = os.path.expanduser("~/.config/spikeon-router/google_token.json")
_CREDS_PATH = os.path.expanduser("~/.config/spikeon-router/google_credentials.json")
# Hermes auth pool
_HERMES_AUTH_PATH = os.path.expanduser("~/.hermes/auth.json")


def _get_gemini_api_key() -> str:
    """Return Gemini API key: env var → Hermes credential pool."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    try:
        with open(_HERMES_AUTH_PATH) as f:
            auth = json.load(f)
        pool = auth.get("credential_pool", {}).get("gemini", [])
        if isinstance(pool, list) and pool:
            return pool[0].get("access_token", "")
    except Exception:
        pass
    return ""

_EMAIL_TERMS = {"email", "gmail", "inbox", "mail", "sent mail"}


def _google_creds():
    # prefer Hermes-managed token, fall back to standalone path
    token = _HERMES_TOKEN_PATH if os.path.exists(_HERMES_TOKEN_PATH) else _TOKEN_PATH
    secret = _HERMES_CREDS_PATH if os.path.exists(_HERMES_CREDS_PATH) else _CREDS_PATH

    creds = None
    if os.path.exists(token):
        creds = Credentials.from_authorized_user_file(token, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif os.path.exists(secret):
            flow = InstalledAppFlow.from_client_secrets_file(secret, SCOPES)
            creds = flow.run_local_server(port=0)
            os.makedirs(os.path.dirname(token), exist_ok=True)
            with open(token, "w") as f:
                f.write(creds.to_json())
        else:
            return None
    return creds


def _fetch_bills() -> str:
    creds = _google_creds()
    if not creds:
        return "(Google OAuth not configured — place credentials.json at ~/.config/spikeon-router/google_credentials.json)"
    try:
        drive = build("drive", "v3", credentials=creds)
        result = drive.files().list(
            q=f"name='{BILLS_SHEET_NAME}' and mimeType='application/vnd.google-apps.spreadsheet'",
            fields="files(id)",
            pageSize=1,
        ).execute()
        files = result.get("files", [])
        if not files:
            return f"(No spreadsheet named '{BILLS_SHEET_NAME}' found in Drive)"
        sheet_id = files[0]["id"]
        sheets = build("sheets", "v4", credentials=creds)
        data = sheets.spreadsheets().values().get(
            spreadsheetId=sheet_id, range="A1:Z500"
        ).execute()
        rows = data.get("values", [])
        if not rows:
            return f"({BILLS_SHEET_NAME} sheet is empty)"
        return f"{BILLS_SHEET_NAME} spreadsheet:\n" + "\n".join(
            "\t".join(str(c) for c in row) for row in rows
        )
    except Exception as e:
        return f"(Bills sheet error: {e})"


def _fetch_gmail(query: str) -> str:
    creds = _google_creds()
    if not creds:
        return ""
    try:
        svc = build("gmail", "v1", credentials=creds)
        result = svc.users().messages().list(userId="me", q=query, maxResults=5).execute()
        msgs = result.get("messages", [])
        if not msgs:
            return "(No matching emails)"
        out = []
        for m in msgs:
            d = svc.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["Subject", "From", "Date"],
            ).execute()
            h = {hdr["name"]: hdr["value"] for hdr in d.get("payload", {}).get("headers", [])}
            snippet = d.get("snippet", "")
            out.append(
                f"[{h.get('Date', '')}] {h.get('Subject', '(no subject)')} "
                f"— from {h.get('From', '')} | {snippet}"
            )
        return "Recent emails:\n" + "\n".join(out)
    except Exception as e:
        return f"(Gmail error: {e})"


def _build_context(prompt: str) -> str:
    p = prompt.lower()
    is_finance = any(w in p for w in FINANCE_KEYWORDS)
    is_email = any(t in p for t in _EMAIL_TERMS)
    parts = []
    if is_finance:
        parts.append(_fetch_bills())
    if is_email:
        parts.append(_fetch_gmail(prompt[:150]))
    return "\n\n".join(p for p in parts if p)


def _to_contents(history: list, prompt: str) -> list:
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        text = msg.get("content") or ""
        if text:
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    return contents


def gemini_chat(prompt: str, system: str, history: list) -> str:
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("No Gemini API key found — run 'hermes auth gemini' or set GEMINI_API_KEY")

    ctx = _build_context(prompt)
    full_system = system + ("\n\n" + ctx if ctx else "")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=full_system,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=_to_contents(history, prompt),
        config=config,
    )
    return response.text


def gemini_stream(prompt: str, system: str, history: list):
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("No Gemini API key found — run 'hermes auth gemini' or set GEMINI_API_KEY")

    ctx = _build_context(prompt)
    full_system = system + ("\n\n" + ctx if ctx else "")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=full_system,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=_to_contents(history, prompt),
        config=config,
    ):
        if chunk.text:
            yield chunk.text
