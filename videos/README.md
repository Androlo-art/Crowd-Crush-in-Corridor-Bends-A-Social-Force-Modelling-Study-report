# Videos

A representative simulation clip per scenario, so a reader can visually confirm the
behaviour described in the thesis. This is a curated subset (not every run has a video).

## Size
The 9 clips total ≈75 MB. They are kept locally in this folder for inspection, but are
**not committed to Git by default** (`.gitignore` excludes `videos/**/*.mp4`) so the
tracked tree stays small (≈31 MB without them). The video files are included in this
public repository for visual inspection of the reported dynamics.

| Section | Clip | Shows |
|---|---|---|
| 01 | `fis_highspeed_clog_illustrative.mp4` | faster-is-slower clogging at high desired speed (illustrative — see section README) |
| 02 | `open_bend_90deg.mp4` | open 90° bend, free draining |
| 02 | `restricted_bend_90deg.mp4` | 90° bend feeding a 1.2 m door (crush) |
| 03 | `corner_sharp_r0.0m.mp4` / `corner_rounded_r2.0m.mp4` | sharp vs rounded inner corner |
| 04 | `coupling_90deg_door1.2m_saturated.mp4` / `…_door2.5m_relieved.mp4` | below vs above the saturation width at 90° |
| 05 | `obstacle_90deg_none.mp4` / `obstacle_90deg_gate.mp4` | 90° bend without vs with the two-pillar gate |
