---
name: run-pipeline
description: Run the FMI pipeline on a video (synthetic, V1 roboflow clip, V2 SoccerTrack half, V3 lower-tier match, club footage) from the CLI or the API, choose config/tier/segment, read stage QA, and troubleshoot failures.
---

# Running the pipeline

## Inputs
- Video: local path, `http(s)://` URL (API downloads it), or `synthetic://seed=1&seconds=20` (fake models).
- Squad CSV: `team,number,name[,position,player_id]`, team = home/away (or the team names).
- Tier: `broadcast` (moving camera), `panoramic` (static 4K, tiling on), `follow_cam`, `static`.

## CLI
```bash
fmi run --match-id v1 --video data/videos/v1_121364_0.mp4 --tier broadcast
fmi run --match-id v2 --video data/soccertrack/<half>.mp4 --tier panoramic --segment 0 600 \
        --squad squad_v2.csv --home-color "#ffffff" --away-color "#0000ff"
fmi run --match-id v2 --config infra/pipeline.example.yaml    # override thresholds
fmi run --match-id v2 --targets ball                          # one stage + deps
fmi run --match-id v2 --force team                            # recompute team and downstream
```
Output: `storage/matches/<id>/package/` (Match Package) + per-stage dirs with `manifest.json` (QA, timing).

## API
`POST /matches` → `POST /matches/{id}/video` (or `/video/import`) → `POST /matches/{id}/squad` →
`POST /matches/{id}/process` → poll `GET /jobs/{job_id}` → `GET /matches/{id}/summary`.

## Reading QA (what to check first)
- `calib.frames_calibrated_frac`, `calib.median_score` (< 0.7 → click 4 points in review)
- `detect.mean_players_per_frame` (< 10 → recall problem: try `detect.imgsz: 1920` or tiling on)
- `track.tracklets` vs `link.long_tracks` (many long tracks → raise `link.max_gap_s`)
- `team.frames_over_11` (> 0 → duplicate tracks: merge in review)
- `identity.player_frames_named_frac` (target ≥ 0.85 automatically)
- `ball.available_frac`, `analytics.impossible_jumps`

## Troubleshooting
- `FileNotFoundError ... weights`: `python scripts/fetch_data.py --models`.
- `ffmpeg/ffprobe not found`: install ffmpeg (ingest needs it for real video).
- Everything recomputes each time: the source file changed (size/mtime) or config differs; check `fmi plan`.
- CUDA OOM on 4K: lower `detect.tile_size` or `detect.imgsz`, keep `-c 1` on the GPU worker.
