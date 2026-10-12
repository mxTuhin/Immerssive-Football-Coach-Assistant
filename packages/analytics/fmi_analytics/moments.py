"""Key-moment candidates for the AI report (8-12 per match), deterministic and explainable.

Each moment carries a score, a reason built from numbers, a time window for the clip and the 3D scene.
The AI analyst picks, orders and narrates from these; it never invents moments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_analytics.common import attack_sign

PRE_S, POST_S = 8.0, 6.0


def _norm_x(row, meta):
    return float(row["start_x"]) * attack_sign(meta, int(row["period"]), row["team"]) if pd.notna(row["start_x"]) else np.nan


def candidate_moments(pkg, events: pd.DataFrame, xt_moves: pd.DataFrame | None = None,
                      sprints: pd.DataFrame | None = None) -> list[dict]:
    meta = pkg.meta
    L = meta.pitch_length
    ev = events.sort_values(["period", "t"]).reset_index(drop=True)
    cands: list[dict] = []

    def add(row, kind, score, reason, team=None, players=None):
        cands.append({"kind": kind, "score": round(float(score), 3), "period": int(row["period"]), "t": float(row["t"]),
                      "team": team or row.get("team"), "players": [p for p in (players or [row.get("player_id")]) if p],
                      "reason": reason, "event_id": row.get("event_id")})

    shots = ev[ev["type"] == "shot"]
    for _, r in shots.iterrows():
        goal = r["outcome"] == "success"
        nx = _norm_x(r, meta)
        dist = L / 2 - nx if np.isfinite(nx) else np.nan
        reason = f"{'Goal' if goal else 'Shot'} by {pkg.player_name(r['player_id'])}" + (
            f" from {dist:.0f} m" if np.isfinite(dist) else "")
        add(r, "goal" if goal else "shot", 1.0 if goal else 0.6 + (0.1 if np.isfinite(dist) and dist < 18 else 0), reason)

    wins = ev[ev["type"].isin(["recovery", "interception", "tackle"])]
    for _, r in wins.iterrows():
        nx = _norm_x(r, meta)
        if not (np.isfinite(nx) and nx > L / 6):
            continue
        nxt = shots[(shots["period"] == r["period"]) & (shots["team"] == r["team"]) & (shots["t"] > r["t"]) &
                    (shots["t"] <= r["t"] + 15)]
        reason = f"Ball won high up the pitch by {pkg.player_name(r['player_id'])} ({nx + L / 2:.0f} m from own goal)"
        if not nxt.empty:
            reason += f", shot {nxt.iloc[0]['t'] - r['t']:.0f} s later"
        add(r, "high_turnover", 0.75 if not nxt.empty else 0.4, reason)

    setp = ev[ev["type"].isin(["corner", "free_kick"])]
    for _, r in setp.iterrows():
        nxt = shots[(shots["period"] == r["period"]) & (shots["t"] > r["t"]) & (shots["t"] <= r["t"] + 10)]
        if not nxt.empty:
            add(r, "set_piece_chance", 0.55, f"{r['type'].replace('_', ' ').title()} leading to a shot")

    if xt_moves is not None and not xt_moves.empty:
        for _, r in xt_moves.nlargest(5, "xt_added").iterrows():
            row = ev[ev["event_id"] == r["event_id"]]
            if row.empty:
                continue
            add(row.iloc[0], "progressive_pass", 0.45 + min(float(r["xt_added"]), 0.3),
                f"Pass by {pkg.player_name(r['player_id'])} adding {r['xt_added']:.3f} xT")

    if sprints is not None and not sprints.empty:
        for _, s in sprints.nlargest(3, "distance_m").iterrows():
            add({"period": s["period"], "t": s["t_start"], "team": s["team"], "player_id": s["player_id"]},
                "sprint", 0.3, f"{pkg.player_name(s['player_id'])} sprint of {s['distance_m']:.0f} m at "
                               f"{s['max_speed_kmh']:.1f} km/h")
    return cands


def select_moments(cands: list[dict], n_min: int = 8, n_max: int = 12, min_gap_s: float = 45.0) -> list[dict]:
    chosen: list[dict] = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if len(chosen) >= n_max:
            break
        if any(o["period"] == c["period"] and abs(o["t"] - c["t"]) < min_gap_s for o in chosen):
            continue
        chosen.append(c)
    if len(chosen) < n_min:  # relax spacing to reach the minimum
        for c in sorted(cands, key=lambda c: -c["score"]):
            if len(chosen) >= n_min:
                break
            if c not in chosen and not any(o["period"] == c["period"] and abs(o["t"] - c["t"]) < 10 for o in chosen):
                chosen.append(c)
    chosen.sort(key=lambda c: (c["period"], c["t"]))
    for i, c in enumerate(chosen):
        c["moment_id"] = f"m{i + 1:02d}"
        c["t_start"] = max(0.0, c["t"] - PRE_S)
        c["t_end"] = c["t"] + POST_S
    return chosen
