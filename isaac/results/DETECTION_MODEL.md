# Fitted detection model: simulator pixels and viewing angle -> real detector (YOLO-World)

Layouts: booth, cubicle, open. There are 17,010 viewpoint–object pairs, and the simulator calls 3,197 of them
visible. The real detector found 85 of the 13,813 pairs the simulator calls invisible (0.6 %).

**Leave-one-layout-out log loss** (lower is better; each row is fitted on the other two layouts):

| Held-out layout | Current single curve | Per group | Per group + viewing angle |
|---|---|---|---|
| booth | 0.6224 | 0.5651 | 0.5627 |
| cubicle | 0.6077 | 0.5502 | 0.5472 |
| open | 0.6125 | 0.5752 | 0.5696 |

**Coefficients** (fitted on all three layouts): p = sigmoid(b0 + b1·ln(sim px) + b2·down/30°), where "down" is how
steeply the camera looks down at the object.

| Group | Classes | b0 | b1 | b2 (per 30° looking down) | Pairs |
|---|---|---|---|---|---|
| tall | cart, chair | -2.476 | 0.517 | -0.627 | 889 |
| medium | bag, bin, box, monitor, plant, stool, toolbox | -2.878 | 0.414 | -0.528 | 1780 |
| flat | laptop | -14.246 | 1.789 | -0.145 | 205 |
| small | mug | -5.069 | 0.939 | 0.155 | 323 |

**Calibration by group and altitude.** Each cell is real detection rate / fitted model / current curve, over pairs
the simulator calls visible:

| Group | 0.4 m | 1 m | 1.8 m |
|---|---|---|---|
| tall | 0.88 / 0.90 / 0.80 | 0.83 / 0.86 / 0.79 | 0.83 / 0.80 / 0.76 |
| medium | 0.54 / 0.56 / 0.68 | 0.54 / 0.53 / 0.70 | 0.46 / 0.45 / 0.65 |
| flat | 0.00 / 0.00 / 0.06 | 0.10 / 0.10 / 0.36 | 0.13 / 0.13 / 0.44 |
| small | 0.18 / 0.26 / 0.16 | 0.50 / 0.45 / 0.31 | 0.44 / 0.44 / 0.28 |
