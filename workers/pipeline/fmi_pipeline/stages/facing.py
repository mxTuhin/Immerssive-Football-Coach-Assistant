"""Stage 10: facing direction (body_yaw) from tracking only; see fmi_analytics.facing for the rules."""

from __future__ import annotations

from fmi_analytics.common import goalkeepers
from fmi_analytics.facing import compute_facing
from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.stages._pkg import _pkg
from fmi_pipeline.stages._util import load, save


class Facing(Stage):
    name = "facing"
    version = "1"
    deps = ("project", "identity")

    def run(self, ctx: StageContext, inputs) -> StageResult:
        pkg = _pkg(inputs["project"])
        ids = load(inputs["identity"], "identities")
        gk = set(ids.loc[ids["role"] == "goalkeeper", "player_id"].dropna()) | set(
            ids.loc[(ids["role"] == "goalkeeper") & ids["player_id"].isna(), "track_id"])
        gk |= set(goalkeepers(pkg).values())
        tr = compute_facing(pkg.tracking, pkg.meta, gk)
        players = tr[tr["team"].isin(["home", "away"])]
        qa = self.qa(players_with_yaw_frac=float(players["body_yaw"].notna().mean()) if len(players) else 0.0,
                     mean_yaw_conf=float(players["yaw_conf"].mean()) if len(players) else 0.0,
                     **{f"source_{k}": float(v) for k, v in players["yaw_source"].value_counts(normalize=True).items()})
        return StageResult({"tracking": save(ctx, tr, "tracking")}, qa)
