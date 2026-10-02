# Stage 3a: calibrating the visibility model on Isaac Sim renders

Rendered on the HP Z2 (RTX 4000 Ada, Isaac Sim 6.0.1), 2026-10-02; calibration run there with
`isaac/calibrate_visibility.py` on the booth, open and cubicle render indexes (360 frames each).

Layouts: booth, open, cubicle. Render-visible = ≥ 200 px in any yaw. For each extended variant, the pixel threshold a50 is fitted on the other layouts and scored on the held-out one (leave-one-layout-out). "Gain" = objects visible per viewpoint at the highest altitude vs the lowest.

| Variant | Held-out layout | a50 (px, fitted on others) | Agreement | κ | Model gain 0.4→1.8 m | Render gain 0.4→1.8 m |
|---|---|---|---|---|---|---|
| point | booth | — | 93.9% | 0.76 | +11% | +41% |
| point | open | — | 91.9% | 0.72 | +44% | +127% |
| point | cubicle | — | 94.3% | 0.80 | +7% | +31% |
| extent | booth | 126 | 95.0% | 0.81 | +19% | +41% |
| extent | open | 126 | 95.7% | 0.85 | +81% | +127% |
| extent | cubicle | 126 | 96.2% | 0.87 | +18% | +31% |
| +vfov | booth | 126 | 95.5% | 0.82 | +10% | +41% |
| +vfov | open | 126 | 96.0% | 0.86 | +81% | +127% |
| +vfov | cubicle | 126 | 96.6% | 0.88 | +12% | +31% |
| +top | booth | 158.7 | 95.6% | 0.83 | +15% | +41% |
| +top | open | 224.5 | 96.3% | 0.87 | +87% | +127% |
| +top | cubicle | 141.4 | 96.8% | 0.89 | +17% | +31% |

**Mean held-out agreement:** point 93.3%, extent 95.6%, +vfov 96.0%, +top 96.2%.

**Chosen:** `+top` with a50 = **224.5 px** fitted on all layouts (agreement 96.3%, κ 0.87). Objects visible per viewpoint by altitude:

| Altitude | Model (calibrated) | Render |
|---|---|---|
| 0.4 m | 12.4 | 11.3 |
| 1 m | 14.7 | 14.7 |
| 1.8 m | 16.9 | 18.3 |

Use in the simulator: `SimConfig(visibility_model="extended", vis_a50_px=224.5)`.

## Interpretation
- Every ingredient improves held-out agreement; the full extended model is best in all three layouts (96.2 % vs 93.3 %).
- The calibrated model still **under-states the altitude gain** (+36 % vs +62 % pooled; +15 % vs +41 % booth,
  +17 % vs +31 % cubicle, +87 % vs +127 % open). Any altitude result computed with it is therefore conservative
  against altitude; the residual is reported alongside the re-run Stage 2 sweep.
