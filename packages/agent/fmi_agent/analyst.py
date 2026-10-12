"""AI analyst: Claude with tools over the Match Package.

* ``ask(question)``: plain-language Q&A, returns the answer, the tool calls made, and a grounding check.
* ``report()``: match report with 8-12 key moments (each linked to a clip window and a 3D scene).

Rules enforced in the prompt and checked afterwards: every number must come from a tool result; data-quality
notes must be surfaced; no player ratings. ``grounding_check`` flags numbers in the answer that no tool returned.

Defaults (see the claude-api skill): model ``claude-opus-5-5``, adaptive thinking, explicit effort, prompt
caching on the stable prefix (tools + system), and the server-side refusal fallback (``fallbacks="default"``)
which is enabled by default and can be turned off with ``FMI_AGENT_FALLBACKS=off``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fmi_agent.prompts import REPORT_INSTRUCTIONS, REPORT_SCHEMA, SYSTEM_PROMPT
from fmi_agent.tools import MatchTools
from fmi_schema.package import MatchPackage

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TURNS = 16

_NUM = re.compile(r"(?<![A-Za-z#])-?\d+(?:[.,]\d+)?%?")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(text):
        pct = m.endswith("%")
        v = float(m.rstrip("%").replace(",", "."))
        out.add(round(v, 2))
        if pct:
            out.add(round(v / 100, 3))
    return out


def grounding_check(answer: str, tool_outputs: list[str], ignore_small: int = 12) -> dict:
    """Numbers in ``answer`` that do not appear (allowing rounding) in any tool output.

    Small integers (<= ``ignore_small``: e.g. "two halves", "top 5", shirt numbers in names) are ignored.
    """
    seen: set[float] = set()
    for o in tool_outputs:
        for v in _numbers(o):
            seen |= {v, round(v), round(v, 1), round(v * 100, 1), round(v * 100), round(v / 1000, 2)}
            if abs(v) >= 60:  # seconds -> minutes:seconds mentions
                seen |= {round(v / 60), int(v // 60), int(v % 60)}
    claimed = {v for v in _numbers(answer) if not (float(v).is_integer() and abs(v) <= ignore_small)}
    ungrounded = sorted(v for v in claimed if v not in seen and round(v) not in seen and round(v, 1) not in seen)
    return {"numbers_claimed": len(claimed), "ungrounded": ungrounded, "ok": not ungrounded}


class MatchAnalyst:
    def __init__(self, pkg: MatchPackage, model: str = "claude-opus-5-5", effort: str = "high",
                 fallbacks: str = "default", client: Any = None, max_tokens: int = 16000):
        self.pkg = pkg
        self.tools = MatchTools(pkg)
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.max_tokens = max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    # ------------------------------------------------------------------ core loop
    def _request(self, messages: list, output_format: dict | None = None):
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=self.tools.specs(),
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort, **({"format": output_format} if output_format else {})},
            cache_control={"type": "ephemeral"},
        )
        if self.fallbacks and self.fallbacks != "off":
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = self.fallbacks
        return self.client.beta.messages.create(**kwargs)

    def _run(self, user_content: str, output_format: dict | None = None) -> dict:
        messages: list = [{"role": "user", "content": user_content}]
        calls: list[dict] = []
        outputs: list[str] = []
        response = None
        for _ in range(MAX_TURNS):
            response = self._request(messages, output_format)
            if response.stop_reason == "refusal":
                return {"answer": None, "refused": True, "tool_calls": calls,
                        "stop_details": getattr(response, "stop_details", None)}
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue
            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                break
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:  # all results go back in ONE user message
                try:
                    out = json.dumps(self.tools.call(tu.name, dict(tu.input or {})), default=str)
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
                    outputs.append(out)
                except Exception as exc:  # report the error to the model instead of failing the turn
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                    "content": f"{type(exc).__name__}: {exc}"})
                calls.append({"name": tu.name, "input": dict(tu.input or {})})
            messages.append({"role": "user", "content": results})
        text = "".join(getattr(b, "text", "") for b in (response.content if response else [])
                       if getattr(b, "type", None) == "text")
        return {"answer": text, "tool_calls": calls, "tool_outputs": outputs,
                "stop_reason": getattr(response, "stop_reason", None),
                "usage": getattr(getattr(response, "usage", None), "model_dump", lambda: None)()}

    # ------------------------------------------------------------------ public API
    def ask(self, question: str) -> dict:
        res = self._run(question)
        if res.get("answer"):
            res["grounding"] = grounding_check(res["answer"], res.get("tool_outputs", []))
            if not res["tool_calls"]:
                res["grounding"]["ok"] = False
                res["grounding"]["reason"] = "no tool call made"
        res.pop("tool_outputs", None)
        return res

    def report(self) -> dict:
        res = self._run(REPORT_INSTRUCTIONS, output_format={"type": "json_schema", "schema": REPORT_SCHEMA})
        if not res.get("answer"):
            return {"error": "no report produced", **{k: v for k, v in res.items() if k != "tool_outputs"}}
        try:
            report = json.loads(res["answer"])
        except json.JSONDecodeError:
            return {"error": "report was not valid JSON", "raw": res["answer"]}
        # attach clip / scene references from the deterministic candidates
        cands = {m["moment_id"]: m for m in self.pkg.derived.get("key_moments", [])}
        for km in report.get("key_moments", []):
            c = cands.get(km.get("moment_id"))
            if c:
                km.update(period=c["period"], t_start=c["t_start"], t_end=c["t_end"],
                          scene=f"/matches/{self.pkg.match_id}/scenes/{c['moment_id']}",
                          clip={"period": c["period"], "t_start": c["t_start"], "t_end": c["t_end"]})
        report["grounding"] = grounding_check(json.dumps(report), res.get("tool_outputs", []))
        report["tool_calls"] = res["tool_calls"]
        report["model"] = self.model
        return report
