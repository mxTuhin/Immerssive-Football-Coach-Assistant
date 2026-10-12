import json
from types import SimpleNamespace

from fmi_agent.analyst import MatchAnalyst, grounding_check
from fmi_agent.tools import MatchTools
from fmi_analytics.run import run_all


def test_tools_return_data(passing_package):
    pkg = run_all(passing_package)
    tools = MatchTools(pkg)
    names = [s["name"] for s in tools.specs()]
    assert "get_match_overview" in names and len(names) == len(set(names))
    ov = tools.call("get_match_overview", {})
    assert ov["teams"]["home"] == "Home FC"
    ev = tools.call("get_events", {"type": "pass", "team": "Home FC"})
    assert ev["count"] >= 4 and ev["events"][0]["clock"].startswith("P1")
    ps = tools.call("get_player_stats", {"player": "Ann"})
    assert ps["players"][0]["player_id"] == "a"
    assert tools.call("get_team_shape", {"team": "home"})["team"] == "home"


def test_grounding_check():
    outs = [json.dumps({"possession": 0.565, "distance_m": 10432.7, "t": 1381.0})]
    ok = grounding_check("Home had 56.5% possession; Ann ran 10432.7 m; at 23:01.", outs)
    assert ok["ok"], ok
    bad = grounding_check("Home completed 87.3% of passes.", outs)
    assert not bad["ok"] and 87.3 in bad["ungrounded"]


class FakeClient:
    """Mimics client.beta.messages.create: first a tool call, then a final answer."""

    def __init__(self):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            block = SimpleNamespace(type="tool_use", id="tu1", name="get_match_overview", input={})
            return SimpleNamespace(stop_reason="tool_use", content=[block], usage=None)
        poss = json.loads(kw["messages"][-1]["content"][0]["content"])["possession_share"]["home"]
        text = SimpleNamespace(type="text", text=f"Home had {poss * 100:.1f}% of the ball.")
        return SimpleNamespace(stop_reason="end_turn", content=[text], usage=None)


def test_analyst_loop_with_fake_client(passing_package):
    pkg = run_all(passing_package)
    client = FakeClient()
    res = MatchAnalyst(pkg, client=client).ask("Who had more possession?")
    assert res["tool_calls"] == [{"name": "get_match_overview", "input": {}}]
    assert res["grounding"]["ok"], res
    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5" and first["thinking"] == {"type": "adaptive"}
    assert first["fallbacks"] == "default" and first["cache_control"] == {"type": "ephemeral"}
