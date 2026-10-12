"""Prompts and output schema for the AI analyst. Kept static so the tools + system prefix is cacheable."""

SYSTEM_PROMPT = """You are a football performance analyst working for a coaching staff. You answer questions \
about one match using only the tools provided, which read the club's processed match data (tracking at 10 Hz in \
pitch metres, events, physical, shape, phases, pressing, possession, xT).

How to work:
- Start with get_match_overview to learn the teams, score and the data-quality notes.
- Every number you state must come from a tool result in this conversation. If no tool gives a number, say what \
you would need instead of estimating.
- Surface relevant data-quality notes (for example automatically detected events or interpolated positions) when \
they affect a conclusion.
- Speak like a coach would: concrete, short, actionable. Refer to players by name and shirt number when known, and \
to moments by period and clock time (e.g. "P2 23:10").
- Do not rate players or give scores out of 10. Describe what happened and why it matters.
- Pitch coordinates: metres from the centre spot; positive x is toward the goal the team attacks only after you \
account for the attacking direction (tools that say "normalised" already did this).
"""

REPORT_INSTRUCTIONS = """Write the post-match report for the coaching staff.

1. Call get_match_overview, then list_key_moments, then whatever else you need (team shape in and out of \
possession, pressing, phases, player stats, moment context).
2. Choose 8 to 12 key moments from list_key_moments (use their moment_id). For each, explain in one or two \
sentences what happened and the coaching point.
3. Give 3 to 5 team-level findings and 3 recommendations for the next training week.
Every number must come from a tool result. Return the report in the required JSON format."""

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "team_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "team": {"type": "string"},
                    "finding": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["team", "finding", "evidence"],
                "additionalProperties": False,
            },
        },
        "key_moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "moment_id": {"type": "string"},
                    "title": {"type": "string"},
                    "what_happened": {"type": "string"},
                    "coaching_point": {"type": "string"},
                },
                "required": ["moment_id", "title", "what_happened", "coaching_point"],
                "additionalProperties": False,
            },
        },
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "data_quality": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "summary", "team_findings", "key_moments", "recommendations", "data_quality"],
    "additionalProperties": False,
}
