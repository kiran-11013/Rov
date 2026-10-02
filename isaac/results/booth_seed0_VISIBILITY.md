# Stage 3a: visibility model vs Isaac Sim render (booth layout, seed 0)

Rendered on the HP Z2 (RTX 4000 Ada, Isaac Sim 6.0.1), 2026-10-02: 360 frames (90 viewpoints × 4 yaws) in 38 s.
Render-visible = at least 200 pixels in any of the yaws at a viewpoint. Model-visible = detection probability ≥ 0.5.

**Pre-registered rule (agreement ≥ 80%): PASS.** Agreement 93.9%, Cohen's κ 0.76, over 5670 viewpoint–object pairs.

| Slice | Pairs | Agreement | κ | Model says visible | Render says visible | Rendered when model says visible | Rendered when model says hidden |
|---|---|---|---|---|---|---|---|
| all | 5670 | 93.9% | 0.76 | 14.1% | 15.3% | 82.7% | 4.3% |
| altitude 0.4 m | 1890 | 93.8% | 0.73 | 13.4% | 12.6% | 74.0% | 3.1% |
| altitude 1 m | 1890 | 94.8% | 0.79 | 13.9% | 15.6% | 87.4% | 4.1% |
| altitude 1.8 m | 1890 | 93.0% | 0.75 | 14.9% | 17.8% | 86.2% | 5.8% |
| class bag | 540 | 94.8% | 0.73 | 10.7% | 10.4% | 74.1% | 2.7% |
| class bin | 360 | 94.2% | 0.74 | 12.2% | 13.6% | 81.8% | 4.1% |
| class box | 720 | 94.7% | 0.84 | 19.7% | 21.1% | 90.1% | 4.2% |
| class cart | 270 | 95.2% | 0.81 | 12.2% | 17.0% | 100.0% | 5.5% |
| class chair | 1080 | 87.7% | 0.63 | 15.7% | 25.6% | 92.4% | 13.2% |
| class laptop | 450 | 96.0% | 0.71 | 9.1% | 6.0% | 61.0% | 0.5% |
| class monitor | 540 | 95.4% | 0.78 | 11.5% | 12.0% | 82.3% | 2.9% |
| class mug | 630 | 96.7% | 0.74 | 8.4% | 5.1% | 60.4% | 0.0% |
| class plant | 270 | 94.1% | 0.80 | 20.0% | 15.6% | 74.1% | 0.9% |
| class stool | 360 | 94.7% | 0.84 | 21.7% | 20.8% | 85.9% | 2.8% |
| class toolbox | 450 | 96.4% | 0.84 | 14.0% | 10.9% | 76.2% | 0.3% |

**Objects visible per viewpoint, by altitude:**

| Altitude | Model | Render |
|---|---|---|
| 0.4 m | 8.5 | 8.0 |
| 1 m | 8.7 | 9.8 |
| 1.8 m | 9.4 | 11.2 |

## Interpretation
- The visibility model used in Stages 1–2 passes the pre-registered 80 % agreement rule.
- It has an altitude-dependent bias: the render sees 40 % more objects per viewpoint at 1.8 m than at 0.4 m, the model
  only 11 %. The model checks one point 0.2 m above each support, so tall objects (chairs) that peek over occluders are
  under-counted, and small low objects (mugs, laptops) are over-counted.
- Because this bias works against altitude, the Stage 2 altitude verdict must be re-checked with a calibrated model
  before it is reported. Next: calibrate on renders from all three layouts, then re-run the Stage 2 layout sweep with
  the same 10 % rule.
