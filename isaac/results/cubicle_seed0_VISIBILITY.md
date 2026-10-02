# Stage 3a: visibility model vs Isaac Sim render (cubicle layout, seed 0)

Rendered on the HP Z2 (RTX 4000 Ada, Isaac Sim 6.0.1), 2026-10-02: 360 frames in 39 s.
Render-visible = at least 200 pixels in any of the yaws at a viewpoint. Model-visible = detection probability ≥ 0.5.

**Pre-registered rule (agreement ≥ 80%): PASS.** Agreement 94.3%, Cohen's κ 0.80, over 5670 viewpoint–object pairs.

| Slice | Pairs | Agreement | κ | Model says visible | Render says visible | Rendered when model says visible | Rendered when model says hidden |
|---|---|---|---|---|---|---|---|
| all | 5670 | 94.3% | 0.80 | 17.7% | 17.0% | 82.1% | 3.0% |
| altitude 0.4 m | 1890 | 93.5% | 0.76 | 17.1% | 14.7% | 74.1% | 2.4% |
| altitude 1 m | 1890 | 94.9% | 0.82 | 17.6% | 17.2% | 84.3% | 2.9% |
| altitude 1.8 m | 1890 | 94.6% | 0.82 | 18.4% | 19.2% | 87.4% | 3.8% |
| class bag | 540 | 95.4% | 0.81 | 15.6% | 12.8% | 76.2% | 1.1% |
| class bin | 360 | 91.1% | 0.74 | 20.6% | 22.2% | 82.4% | 6.6% |
| class box | 720 | 93.9% | 0.80 | 18.2% | 20.1% | 88.5% | 4.9% |
| class cart | 270 | 95.9% | 0.87 | 16.7% | 20.7% | 100.0% | 4.9% |
| class chair | 1080 | 95.0% | 0.87 | 23.0% | 26.9% | 97.6% | 5.8% |
| class laptop | 450 | 94.2% | 0.70 | 13.3% | 8.0% | 58.3% | 0.3% |
| class monitor | 540 | 95.7% | 0.83 | 13.3% | 15.4% | 91.7% | 3.6% |
| class mug | 630 | 91.9% | 0.53 | 13.3% | 5.2% | 39.3% | 0.0% |
| class plant | 270 | 92.2% | 0.75 | 23.3% | 15.6% | 66.7% | 0.0% |
| class stool | 360 | 96.4% | 0.89 | 22.8% | 20.8% | 87.8% | 1.1% |
| class toolbox | 450 | 95.1% | 0.79 | 13.6% | 12.7% | 78.7% | 2.3% |

**Objects visible per viewpoint, by altitude:**

| Altitude | Model | Render |
|---|---|---|
| 0.4 m | 10.8 | 9.3 |
| 1 m | 11.1 | 10.8 |
| 1.8 m | 11.6 | 12.1 |

## Interpretation
Same pattern as the booth layout: passes the rule, but the model over-counts at 0.4 m and under-counts the gain from
altitude (render +30 % from 0.4 m to 1.8 m, model +7 %). Small objects are the weakest point (mugs rendered in only
39 % of model-visible cases).
