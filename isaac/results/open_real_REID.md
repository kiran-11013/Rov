# Stage 3b: re-identification with DINOv2 and a depth size check (open layout, seed 0)

The data are the frames and YOLO-World detections in `open_real_DETECTION.md`.

- **Memory:** a crop of each object (63 in total) from its clearest frame.
- **Queries:** 1,029 detection boxes (confidence ≥ 0.05) from other viewpoints, each overlapping exactly one object.
- **Embedding:** `dinov2_vits14`, compared by cosine similarity.

Objects of the same class are the *same model*, except mugs (4 designs). So same-class objects are true look-alikes.

## Appearance

| Measure | Result |
|---|---|
| Most similar memory is the right object | 16.7 % (chance 1.6 %) |
| Most similar memory is the right class | 70.8 % |
| Within its own class, the right object is most similar | 25.0 % |
| AUC, right object vs other-class memories | 0.880 |
| AUC, right object vs same-class look-alikes | 0.571 |

Appearance separates classes reasonably well, but cannot tell identical look-alikes apart. This is the premise of the
simulator's look-alike experiments. There, the belief, used as a spatial prior, decides between look-alikes, which
adds 6–15 points of success.

## Size from depth

Metric size is the box height (or its largest side, for flat classes) × the 30th-percentile depth in the box ÷ focal
length.

**Plausibility check** (within a factor 2 of the class size): correct labels pass 92 % of the time, wrong labels 68 %.

**Nearest-size rule:** of the detector's label and the true class, which one does the measured size fit better? The
size picks the true class in **73 %** of 278 confusions:

| True class | Detector's wrong label | Queries | Size picks the true class |
|---|---|---|---|
| stool | chair | 59 | 86 % |
| monitor | laptop | 59 | 93 % |
| chair | monitor | 34 | 12 % |
| monitor | box | 23 | 65 % |
| monitor | chair | 20 | 100 % |
| bin | mug | 16 | 100 % |
| bag | mug | 11 | 100 % |
| cart | chair | 10 | 60 % |

The failure case is chair → "monitor": the detector boxes only the chair's backrest. A partial box measures a partial
size, so size cannot recover the true class there.

## What this means for the pipeline

- Detection gives proposals.
- Size removes most scale confusions.
- Appearance narrows a proposal to the right class (AUC 0.88).
- The map belief decides between identical look-alikes.

The simulator models these steps as one appearance-similarity draw. The next step is to calibrate that draw to these
numbers, and check whether the Stage 2 conclusions still hold.
