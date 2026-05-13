---
name: Testing philosophy — prompts first
description: User's rule for how tests must be written in this project
type: feedback
---

Tests must be written from real user prompts, never written to make the tests pass.

**Why:** Writing tests to satisfy code paths produces green tests that prove nothing about real behavior. The test suite should document what a real user would type and what outcome they'd expect — not what input makes a given branch execute.

**How to apply:**
- Every test case must start with a realistic human prompt (e.g. "my login button isn't doing anything", not "fix error" because "error" is a CODE_KEYWORD)
- Never reverse-engineer a prompt from the implementation
- If a routing rule can't be illustrated with a real-world prompt, question whether the rule is right
- Seed new tests from actual usage examples when possible
