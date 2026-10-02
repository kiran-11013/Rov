# Stage 3b: YOLO-World on Isaac Sim renders with realistic models (open layout, seed 0)

360 frames: 30 viewpoints × 3 altitudes × 4 yaws. The 63 objects are realistic library models (`isaac/assets.json`),
each fitted inside its class proxy. The detector is `yolov8x-worldv2.pt` with 28 prompts for 11 classes. The
toolbox class is a briefcase model, because the library has no toolbox. The operating confidence is 0.05: the lowest
setting with ≤ 0.5 background false positives per frame (0.41 here).

- **Detected:** a box with the right class label matches the object.
- **Proposed:** a box of roughly the object's size overlaps it, whatever its label.

## Detection vs size

| Visible pixels | Pairs | Detected | Proposed |
|---|---|---|---|
| 1–50 | 91 | 2.2 % | 2.2 % |
| 50–200 | 241 | 24.9 % | 31.1 % |
| 200–800 | 283 | 41.7 % | 54.1 % |
| 800–3200 | 274 | 73.7 % | 86.5 % |
| 3200–12800 | 230 | 84.3 % | 94.8 % |
| ≥ 12800 | 95 | 73.7 % | 96.8 % |

Fitted p(detect | px) = p_max · logistic((ln px − ln a50) / slope):

| Curve | p_max | a50 (px) | slope |
|---|---|---|---|
| Real detector, right label | 0.85 | 339.6 | 1.0 |
| Real detector, any label (proposals) | 0.99 | 272.4 | 1.0 |
| Simulator, calibrated on render pixels (Stage 3a) | 0.95 | 224.5 | 0.35 |

The real detector needs about 1.5× more pixels than the simulator assumed. Its curve is also much shallower: some
small objects are found, and some large ones are missed.

## Altitude survives a real detector

| Altitude | Visible (≥ 200 px) / viewpoint | Detected / viewpoint | Proposed / viewpoint |
|---|---|---|---|
| 0.4 m | 7.1 | 5.2 | 6.2 |
| 1.0 m | 9.5 | 7.1 | 8.4 |
| 1.8 m | 12.8 | 9.3 | 11.3 |
| gain 0.4 → 1.8 m | +80 % | +79 % | +82 % |

Mugs, the class behind most of the simulated altitude saving: 2.9 / 11.9 / 14.3 % detected at 0.4 / 1.0 / 1.8 m.
Closed laptops seen from above are the detector's weak spot: 0 / 2.7 / 4.7 %. When one is clearly visible
(≥ 1600 px), it gets no box at all 43 % of the time.

## Failure mode: labels, not sight

How clearly visible objects (≥ 1600 px) were labelled:

| Class | Labels |
|---|---|
| trash can | cup 63 %, trash can 19 % |
| stool | chair 77 %, stool 9 % |
| cart | chair 48 %, trolley 29 % |
| monitor | tv 37 %, laptop 35 % |
| laptop | nothing 43 %, laptop / closed laptop 47 % |
| chair, box, plant | correct ≥ 82 % |

The open-vocabulary detector has no sense of scale, so a trash can is a "cup" and a stool is a "chair". Proposals
are still found: ≥ 87 % for every class except laptop, bag and briefcase. So detection should be treated as
proposals, and identity left to re-identification, i.e. appearance matching against the remembered object plus a
metric size check from depth.

## Caveats

- One layout, one seed, rendered scenes.
- Each model is a single asset per class (four for mugs), so the detector sees little within-class variety.
- Furniture is still plain boxes.
