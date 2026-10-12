"""Stage 7: player identity per track.

1. Jersey numbers: pretrained PARSeq on torso/back crops, votes weighted by confidence across each track.
2. Number + team -> squad list player. Two tracks that overlap in time cannot be the same player: the one
   with fewer votes is unassigned and sent to review (the team constraint blocks cross-team swaps).
3. Unnamed tracks inherit an identity by pretrained OSNet ReID similarity to named tracks of the same team
   when they never overlap in time with that player.
4. Squad slots: a long unnamed track becomes the only unseen starter of its team (low confidence).
5. Human ``assign`` corrections override everything.
Unresolved tracks keep a track id (``t<id>``) and appear in the review queue sorted by confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.stages._util import crops_for_tracks, load, proxy, save
from fmi_schema.models import ReviewItem


def vote_numbers(reads: list[tuple[str, float]], min_share: float, min_votes: float) -> tuple[int | None, float]:
    tally: dict[str, float] = {}
    for text, conf in reads:
        if text and text.isdigit() and 1 <= int(text) <= 99:
            tally[text] = tally.get(text, 0.0) + conf
    if not tally:
        return None, 0.0
    total = sum(tally.values())
    best = max(tally, key=tally.get)
    share = tally[best] / total
    if share < min_share or tally[best] < min_votes:
        return None, share * min(1.0, tally[best] / max(min_votes, 1e-9))
    return int(best), float(min(1.0, share * min(1.0, tally[best] / (2 * min_votes))))


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


class Identity(Stage):
    name = "identity"
    version = "1"
    deps = ("ingest", "link", "team")
    corrections = ("assign",)
    config_sections = ("identity",)

    def run(self, ctx: StageContext, inputs) -> StageResult:
        cfg = ctx.config.identity
        roster = load(inputs["ingest"], "roster")
        tracks = load(inputs["link"], "tracks")
        teams = load(inputs["team"], "teams")
        npz = np.load(inputs["link"].path("embeddings"))
        reid_embs = {k: npz[k] for k in npz.files}
        span = tracks.groupby("track_id")["frame"].agg(["min", "max"])
        video = proxy(inputs["ingest"])
        ocr = ctx.models.ocr if ctx.models.has("ocr") else None
        persons = teams[teams["team"].isin(["home", "away"])]
        reads: dict[str, list] = {}
        if ocr is not None and len(persons):
            crops = crops_for_tracks(video, tracks[tracks["track_id"].isin(persons["track_id"])],
                                     cfg.crops_per_track, region=(0.1, 0.6))
            for tid, cs in crops.items():
                reads[tid] = ocr.read(cs)
        rows = []
        for r in teams.itertuples(index=False):
            num, conf = vote_numbers(reads.get(r.track_id, []), cfg.min_vote_share, cfg.min_votes)
            rows.append({"track_id": r.track_id, "team": r.team, "role": r.role, "number": num, "number_conf": conf,
                         "player_id": None, "id_conf": 0.0, "method": None,
                         "start": int(span.loc[r.track_id, "min"]) if r.track_id in span.index else 0,
                         "end": int(span.loc[r.track_id, "max"]) if r.track_id in span.index else 0})
        ids = pd.DataFrame(rows)
        review: list[ReviewItem] = []
        if not ids.empty and not roster.empty:
            lookup = {(t, int(n)): pid for t, n, pid in roster[["team", "number", "player_id"]].dropna().itertuples(index=False)}
            for i, r in ids.iterrows():
                if r["number"] is not None and not pd.isna(r["number"]):
                    pid = lookup.get((r["team"], int(r["number"])))
                    if pid:
                        ids.loc[i, ["player_id", "id_conf", "method"]] = [pid, r["number_conf"], "jersey"]
            # temporal conflicts: same player on overlapping tracks
            for pid, g in ids[ids["player_id"].notna()].groupby("player_id"):
                g = g.sort_values("number_conf", ascending=False)
                kept: list[tuple[int, int]] = []
                for i, r in g.iterrows():
                    if any(overlaps((r["start"], r["end"]), k) for k in kept):
                        ids.loc[i, ["player_id", "id_conf", "method"]] = [None, 0.0, None]
                        review.append(ReviewItem(item_id=f"id-conflict-{r['track_id']}", stage=self.name, kind="identity",
                                                 priority=0.2, frame=int(r["start"]), track_ids=[r["track_id"]],
                                                 message=f"jersey says {pid} but that player is already on another track"))
                    else:
                        kept.append((r["start"], r["end"]))
            # ReID inheritance
            named = ids[ids["player_id"].notna()]
            for i, r in ids[ids["player_id"].isna() & ids["team"].isin(["home", "away"])].iterrows():
                e = reid_embs.get(r["track_id"])
                if e is None:
                    continue
                best, best_sim = None, cfg.reid_link_sim
                for pid, g in named[named["team"] == r["team"]].groupby("player_id"):
                    if any(overlaps((r["start"], r["end"]), (s, t)) for s, t in zip(g["start"], g["end"])):
                        continue
                    sims = [float(e @ reid_embs[t]) for t in g["track_id"] if t in reid_embs]
                    if sims and max(sims) > best_sim:
                        best, best_sim = pid, max(sims)
                if best:
                    ids.loc[i, ["player_id", "id_conf", "method"]] = [best, 0.5 * best_sim, "reid"]
            # squad slots: only one unseen starter left in a team -> the long unnamed track of that team
            for team in ("home", "away"):
                starters = set(roster[(roster["team"] == team) & roster["starter"].fillna(True).astype(bool)]["player_id"])
                seen = set(ids.loc[ids["team"] == team, "player_id"].dropna())
                unseen = starters - seen
                cand = ids[(ids["team"] == team) & ids["player_id"].isna() & (ids["role"] != "referee")]
                if len(unseen) == 1 and len(cand):
                    j = (cand["end"] - cand["start"]).idxmax()
                    ids.loc[j, ["player_id", "id_conf", "method"]] = [next(iter(unseen)), 0.4, "slot"]
        # human assignments win
        assign = ctx.corrections.get("assign") or {}
        for tid, pid in assign.items():
            m = ids["track_id"] == str(tid)
            ids.loc[m, ["player_id", "id_conf", "method"]] = [pid, 1.0, "human"]
        # review queue for players without confident identity, longest first
        for r in ids[(ids["role"] != "referee") & (ids["id_conf"] < cfg.review_conf)].itertuples():
            dur = r.end - r.start
            review.append(ReviewItem(item_id=f"id-{r.track_id}", stage=self.name, kind="identity",
                                     priority=float(np.clip(r.id_conf, 0, 1) * 0.5 + 0.5 / (1 + dur / 250)),
                                     frame=int(r.start), track_ids=[r.track_id],
                                     message=f"assign player ({r.team}, current: {r.player_id or 'unknown'}, "
                                             f"method {r.method})"))
        frames_total = (ids["end"] - ids["start"] + 1).clip(lower=0)
        auto_named = ids["player_id"].notna() & (ids["method"] != "human") & (ids["role"] != "referee")
        qa = self.qa(tracks=len(ids), named=int(ids["player_id"].notna().sum()),
                     named_by_jersey=int((ids["method"] == "jersey").sum()),
                     named_by_reid=int((ids["method"] == "reid").sum()),
                     named_by_slot=int((ids["method"] == "slot").sum()),
                     human=int((ids["method"] == "human").sum()),
                     player_frames_named_frac=float(frames_total[auto_named].sum() /
                                                    max(frames_total[ids["role"] != "referee"].sum(), 1)),
                     ocr=bool(ocr is not None))
        qa.review = sorted(review, key=lambda r: r.priority)[:300]
        return StageResult({"identities": save(ctx, ids, "identities")}, qa)
