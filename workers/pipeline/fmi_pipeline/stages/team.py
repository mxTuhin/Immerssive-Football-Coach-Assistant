"""Stage 6: team and role per track.

SigLIP embeddings of torso crops -> UMAP -> KMeans(2) on outfield players (as in roboflow/sports).
Goalkeepers (detector class) join the team whose players stand on their side of the pitch.
Referees come from the detector class and from colour outliers far from both team centroids.
Cluster -> home/away naming uses the kit colours from the match input when given (else a review item).
The "<= 11 per team on the pitch" rule is checked here and violations go to review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.models.base import CLASS_IDS
from fmi_pipeline.stages._util import crops_for_tracks, load, proxy, save
from fmi_schema.models import ReviewItem


def reduce_dims(X: np.ndarray, dims: int) -> np.ndarray:
    if len(X) <= dims + 2:
        return X
    try:
        import umap  # type: ignore

        return umap.UMAP(n_components=dims, random_state=0, n_neighbors=min(15, len(X) - 1)).fit_transform(X)
    except Exception:
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:dims].T


def kmeans2(X: np.ndarray, seed: int = 0, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import KMeans  # type: ignore

        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X)
        return km.labels_, km.cluster_centers_
    except Exception:
        rng = np.random.default_rng(seed)
        # farthest-point init
        c0 = X[rng.integers(len(X))]
        c1 = X[np.argmax(((X - c0) ** 2).sum(1))]
        C = np.stack([c0, c1])
        for _ in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), 1)
            newC = np.stack([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(2)])
            if np.allclose(newC, C):
                break
            C = newC
        return lab, C


def hex_to_bgr(h: str) -> np.ndarray:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], float)


def mean_torso_bgr(crops: list) -> np.ndarray:
    vals = []
    for c in crops:
        a = np.asarray(c)
        if a.ndim != 3 or a.size == 0:
            continue
        h, w = a.shape[:2]
        t = a[int(0.15 * h):int(0.55 * h), int(0.2 * w):int(0.8 * w)].reshape(-1, 3)
        if len(t):
            vals.append(np.median(t, 0))
    return np.median(vals, 0) if vals else np.full(3, np.nan)


class Team(Stage):
    name = "team"
    version = "1"
    deps = ("ingest", "link")
    corrections = ("team",)
    config_sections = ("team",)
    uses_source = True  # kit colours

    def run(self, ctx: StageContext, inputs) -> StageResult:
        cfg = ctx.config.team
        mi = ctx.match_input
        tracks = load(inputs["link"], "tracks")
        video = proxy(inputs["ingest"])
        crops = crops_for_tracks(video, tracks, cfg.crops_per_track)
        emb_model = ctx.models.team_embedder
        info = []
        for tid, g in tracks.groupby("track_id", sort=False):
            votes = g["cls"].value_counts()
            c = crops.get(str(tid), [])
            e = emb_model.embed(c) if c else np.zeros((0, emb_model.dim))
            info.append({"track_id": str(tid), "n": len(g), "cls": int(votes.idxmax()),
                         "cls_share": float(votes.max() / votes.sum()),
                         "emb": e.mean(0) if len(e) else np.full(emb_model.dim, np.nan),
                         "bgr": mean_torso_bgr(c), "mean_px": float(g["px"].mean())})
        df = pd.DataFrame(info)
        qa = self.qa()
        review: list[ReviewItem] = []
        df["team"], df["role"], df["team_conf"] = "unknown", "player", 0.0
        df.loc[df["cls"] == CLASS_IDS["referee"], ["team", "role", "team_conf"]] = ["referee", "referee", 0.8]
        df.loc[df["cls"] == CLASS_IDS["goalkeeper"], "role"] = "goalkeeper"
        players = df[(df["role"] == "player") & df["emb"].map(lambda v: np.isfinite(v).all())]
        if len(players) >= 2:
            X = np.stack(players["emb"].to_numpy())
            Z = reduce_dims(X, cfg.umap_dims)
            lab, C = kmeans2(Z)
            d = np.linalg.norm(Z[:, None] - C[None], axis=2)
            conf = np.clip(1 - d.min(1) / np.maximum(d.max(1), 1e-9), 0, 1)
            # colour outliers far from both centroids look like referees
            z = (d.min(1) - d.min(1).mean()) / max(d.min(1).std(), 1e-9)
            outlier = (z > cfg.referee_outlier_z) & (players["n"].to_numpy() > 25)
            # name clusters
            names = {0: "home", 1: "away"}
            named_by_colour = False
            if mi.home_color and mi.away_color:
                cb = [np.nanmedian(np.stack(players["bgr"].to_numpy()[lab == k]), 0) for k in (0, 1)]
                hc, ac = hex_to_bgr(mi.home_color), hex_to_bgr(mi.away_color)
                if all(np.isfinite(c).all() for c in cb):
                    cost_keep = np.linalg.norm(cb[0] - hc) + np.linalg.norm(cb[1] - ac)
                    cost_swap = np.linalg.norm(cb[0] - ac) + np.linalg.norm(cb[1] - hc)
                    names = {0: "home", 1: "away"} if cost_keep <= cost_swap else {0: "away", 1: "home"}
                    named_by_colour = True
            if not named_by_colour:
                review.append(ReviewItem(item_id="team-naming", stage=self.name, kind="team", priority=0.2,
                                         message="confirm which cluster is the home team (no kit colours given)"))
            idx = players.index
            df.loc[idx, "team"] = [names[k] for k in lab]
            df.loc[idx, "team_conf"] = conf
            df.loc[idx[outlier], ["team", "role", "team_conf"]] = ["referee", "referee", 0.5]
        # goalkeepers: the team whose outfield players are on the same side
        side = df[df["team"].isin(["home", "away"]) & (df["role"] == "player")].groupby("team")["mean_px"].mean()
        for i, r in df[df["role"] == "goalkeeper"].iterrows():
            if len(side) == 2 and np.isfinite(r["mean_px"]):
                df.loc[i, ["team", "team_conf"]] = [(side - r["mean_px"]).abs().idxmin(), 0.7]
        # human fixes win
        fixes = ctx.corrections.get("team") or {}
        for tid, team in fixes.items():
            m = df["track_id"] == str(tid)
            df.loc[m, ["team", "team_conf"]] = [team, 1.0]
            if team == "referee":
                df.loc[m, "role"] = "referee"
        for r in df[(df["team_conf"] < 0.3) & (df["n"] > 25)].itertuples():
            review.append(ReviewItem(item_id=f"team-{r.track_id}", stage=self.name, kind="team",
                                     priority=0.3 + r.team_conf, track_ids=[r.track_id],
                                     message=f"uncertain team ({r.team}, conf {r.team_conf:.2f})"))
        # <= 11 per team on the pitch
        t = tracks.merge(df[["track_id", "team"]], on="track_id")
        counts = t[t["team"].isin(["home", "away"])].groupby(["frame", "team"])["track_id"].nunique()
        over = counts[counts > 11]
        if len(over):
            frames = over.index.get_level_values(0)
            review.append(ReviewItem(item_id="team-over11", stage=self.name, kind="merge", priority=0.25,
                                     frame=int(frames[0]),
                                     message=f"> 11 players of one team in {frames.nunique()} frames "
                                             "(duplicate tracks: merge them)"))
        out = df[["track_id", "team", "role", "team_conf", "n", "cls_share"]]
        qa.metrics.update(tracks=len(df), home=int((df["team"] == "home").sum()),
                          away=int((df["team"] == "away").sum()), referees=int((df["role"] == "referee").sum()),
                          frames_over_11=int(over.index.get_level_values(0).nunique()) if len(over) else 0,
                          human_fixes=len(fixes))
        qa.review = review
        return StageResult({"teams": save(ctx, out, "teams")}, qa)
