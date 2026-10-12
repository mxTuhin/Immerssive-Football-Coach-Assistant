---
name: ai-analyst
description: Extend or debug the FMI AI analyst (Claude + deterministic tools over the Match Package) - add tools, change prompts or the report schema, keep every number grounded in tool output, and run the 30-question eval.
---

# AI analyst

Code: `packages/agent/fmi_agent/` — `tools.py` (MatchTools + JSON schemas), `analyst.py` (manual tool loop on
`client.beta.messages.create`, grounding check), `prompts.py` (system prompt, report instructions + JSON schema),
`evaluate.py` + `eval/questions.yaml`.

## Defaults (keep unless the user decides otherwise)
`claude-opus-5-5`, `thinking={"type": "adaptive"}`, `output_config={"effort": "high"}`, top-level
`cache_control={"type": "ephemeral"}` (tools + system are a static prefix — never put per-match or time-varying
text in the system prompt), server-side refusal fallback `betas=["server-side-fallback-2026-07-01"],
fallbacks="default"` (turn off with `FMI_AGENT_FALLBACKS=off`). Forced `tool_choice` is not supported on this
model: steer via the prompt. Check `stop_reason` (`refusal`, `pause_turn`) before reading content. Return all
tool results in one user message.

## Rules
- Every number in an answer must come from a tool result: `grounding_check` flags the rest; the eval fails on any.
- Tools are deterministic, return units/definitions and data-quality `notes`; no player ratings.
- Report moments must use `moment_id`s from `list_key_moments` (clip + scene links are attached afterwards).

## Add a tool
1. Method on `MatchTools` returning a JSON-serialisable dict (use `_round`, `_format_events`, `_notes`).
2. Spec in `MatchTools.specs()` (`additionalProperties: false`; keep order stable for caching).
3. Test in `tests/test_agent.py` (fake client pattern: no API calls in CI).

## Run
`POST /matches/{id}/ask {"question": ...}`, `POST /matches/{id}/report`, or
`python -m fmi_agent.evaluate storage/matches/<id>/package` (needs `ANTHROPIC_API_KEY`; costs credits — confirm
with the user before running the full set).
