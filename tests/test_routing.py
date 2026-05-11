"""
Tests for prompt routing decisions.

Each test case is a realistic user prompt with an assertion about which
type of model should handle it. Tests are never written to satisfy a
keyword — they're written from the user's perspective.
"""
import pytest
from router import classify_prompt


# --- Finance / Google Workspace → Gemini ---

def test_asking_about_monthly_bills_goes_to_gemini():
    assert classify_prompt("how much did I spend on subscriptions last month") == "gemini"


def test_asking_about_rent_goes_to_gemini():
    assert classify_prompt("is my rent payment overdue") == "gemini"


def test_asking_to_check_email_goes_to_gemini():
    assert classify_prompt("can you check my inbox for anything from my landlord") == "gemini"


def test_asking_about_google_sheet_goes_to_gemini():
    assert classify_prompt("pull up my google sheet and tell me the totals") == "gemini"


# --- Code / Debug → Coder ---

def test_asking_to_write_a_function_goes_to_coder():
    assert classify_prompt("write me a Python function that validates an email address") == "coder"


def test_asking_why_code_crashes_goes_to_coder():
    assert classify_prompt("my JavaScript keeps throwing a TypeError, can you help debug it") == "coder"


def test_asking_for_a_sql_query_goes_to_coder():
    assert classify_prompt("write a SQL query to find all users who haven't logged in for 30 days") == "coder"


# --- Reasoning / Analysis → Thinker ---

def test_asking_which_option_is_better_goes_to_thinker():
    assert classify_prompt("should I use PostgreSQL or MongoDB for a social app") == "thinker"


def test_asking_for_tradeoffs_goes_to_thinker():
    assert classify_prompt("what are the tradeoffs between microservices and a monolith") == "thinker"


def test_asking_to_analyze_a_situation_goes_to_thinker():
    assert classify_prompt("analyze whether it makes sense to rewrite this in Rust") == "thinker"


# --- Writing / Explanation → Balanced ---

def test_asking_for_an_explanation_goes_to_balanced():
    assert classify_prompt("explain what a foreign key constraint is like I'm new to databases") == "balanced"


def test_asking_to_summarize_goes_to_balanced():
    assert classify_prompt("summarize what we talked about in this conversation") == "balanced"


def test_asking_to_write_an_essay_goes_to_balanced():
    assert classify_prompt("write a short essay on why open source matters") == "balanced"


# --- Short / Trivial → Fast ---

def test_short_factual_question_goes_to_fast():
    assert classify_prompt("what does DNS stand for") == "fast"


def test_greeting_goes_to_fast():
    assert classify_prompt("hey what's up") == "fast"
