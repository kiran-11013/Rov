# Look Before You Fly — simulation of the complete algorithm

This folder implements every algorithmic part of `research/PROPOSAL.md` (v7) in simulation, before any hardware work:
six weeks of object movement, the persistence priors (E1), the trust / verify / search policy with aerial
viewpoints, re-identification, and all E2 arms including the fixed-height ablation, plus grounding of
language commands with conformal clarification (E3).

```bash
pip install -r requirements.txt
python run_all.py            # both layouts, about 40 s → results/{open,cubicle}/RESULTS.md + figures
python run_all.py --quick    # smoke run
python run_stage2.py         # Stage 2 stress tests, about 16 min on 4 cores → results/stage2/STAGE2.md
python -m pytest             # 21 tests
```

## What is simulated

| Proposal part | Module | How |
|---|---|---|
| Three connected spaces with occluders | `world.py` | 18 × 8 m: lab (desks, 1.3–1.4 m partitions, tall shelf), corridor, store (2 m shelves). Two layouts: `open` (free-standing partitions) and `cubicle` (desks enclosed on three sides). 119 resting places on floor, tables and shelves. |
| Six-week tracked ground truth (E1 data) | `dynamics.py` | 64 objects in 11 classes. Time-to-move is Weibull in *activity hours* (weekday 09:00–19:00). Bags and laptops leave in the evening and return in the morning. Moves go mostly to the same room. |
| Persistence priors | `priors.py` | Global constant · persistence filter (exponential, wall clock) · commonsense prior (**LLM stand-in**) · 10 human guessers (**stand-in**) · **ours**: per-class Weibull on activity exposure, shrunk toward the pooled fit, plus an empirical return probability. |
| Where a moved object went | `belief.py` | Learned room-transition table per class (Dirichlet-smoothed), uniform within the room; separate "absent" mass. |
| Seeing from a viewpoint | `geometry.py`, `drone.py` | 3D line of sight: walls block at any height, furniture blocks below its top. Viewpoints = reachable grid nodes × {0.4, 1.0, 1.8} m. Time = path length / 1 m/s + climb / 0.5 m/s; energy = 180 W hover + climb work; cost = time + λ·energy. |
| Trust / verify / search | `policy.py` | Each action is a choice of first viewpoint, followed by belief-greedy search (probability seen per unit cost). Expected costs come from rolling that rule forward on the belief. Ours picks the cheapest option. |
| Re-identification | `perception.py` | Noisy appearance embeddings. Identical-looking chairs pass the appearance threshold together; the belief acts as a spatial prior to break ties. |
| E2 command trials | `episode.py`, `experiments.py` | Commands 1 h and 1 day after mapping. All arms run on the same commands with common random numbers. Success = right instance within 1 m inside the budget. Object-cluster bootstrap CIs. Pre-registered exact McNemar test with Holm correction for the secondary tests. |
| E3 grounding | `grounding.py` | Referring expressions ("the red bag near the store door") generated from the world *at command time*. Map-only vs staleness-aware resolver. Split-conformal sets; the drone asks when the set has more than one ID. |

**Stand-ins, stated plainly:** the commonsense and human priors use hand-set medians. They are not real LLM, Jev or human outputs. Real models plug into the same `Prior.predict` and resolver `scores` interfaces. All numbers in `results/` come from the simulator.

## What the simulation says (seed 0; see `results/*/RESULTS.md`)

1. **E1 – priors.**
   - Our activity-aware Weibull prior has the best Brier score (0.109 overall). It is tied with the persistence filter within the CIs and clearly beats the commonsense / human stand-ins (0.147) and the global constant (0.167).
   - The gap opens at **1 day** (0.166 vs 0.242). Commonsense guesses can't capture overnight returns.
   - Three classes (monitor, bin, plant) have fewer than 15 move events and are pooled, exactly as the audit predicted.
2. **E2 – policy.**
   - The cost-aware policy is **as successful** as trust-then-search (0.94 vs 0.94; McNemar p = 0.45 open, 0.73 cubicle).
   - It is **slightly faster**: −0.7 s in the cubicle layout [CI −1.2, −0.2]; −0.5 s in the open layout, but that CI [−1.0, +0.1] crosses zero.
   - On objects that actually moved, it is **18–22 % faster** (open: 22.1 vs 25.2 s mean, median 14.3 vs 20.5 s).
   - In these layouts it is statistically indistinguishable from **verify-always**.
3. **Altitude.**
   - In both layouts, flying at 0.4 m loses nothing measurable once the policy chooses viewpoints well (Δ −0.1 to +0.25 s).
   - Altitude helps only the naive trust baseline (≈ −0.4 s).
   - Cubicles open to the aisle can be looked into from 0.4 m, and climbing costs time.
   - **This is the pre-registered "verify-always / altitude doesn't matter" outcome.** The real study needs layouts where low views are genuinely blocked (cubicles facing walls, high shelves, cluttered tables), or it should lead with the benchmark.
4. **Re-identification dominates failures.**
   - Every instance-level failure is an identical chair.
   - Using the belief as a spatial prior raised the policy's success from 0.895 to 0.935.
   - A **miscalibrated prior also hurts re-ID**: the commonsense stand-in loses 3 points of success (Holm p ≤ 0.01) because it pushes the drone to accept a twin elsewhere.
5. **E3 – grounding.**
   - The staleness-aware resolver is slightly more accurate (argmax 0.95 vs 0.94; accuracy when it acts 0.99 vs 0.97–0.98) at the same conformal coverage (≈ 0.92 for α = 0.1).
   - Ask rates are high (≈ 0.6) because many generated references are genuinely ambiguous; 96–97 % of ambiguous references trigger a question.

## Implications for the real study

- The **prior benchmark (E1)** and **re-ID under identical objects** carry the clearest signal. Budget effort there.
- Design E2 spaces so altitude *can* matter: wall-facing cubicles, shelf tops above 1.5 m, clutter on desks. Otherwise expect the null result above.
- Report instance-exact and intent-equivalent success separately: they differ only on identical objects.
- Calibration of the prior matters twice, once for the action and once for re-ID. That is an argument for the E1 benchmark.

## Stage 2: stress tests (see `results/stage2/STAGE2.md`)

Stage 1 used one seed and two layouts. Stage 2 checks whether its conclusions survive:
- a **layout sweep**: 20 layouts × 6 seeds, covering partition height 1.0–2.0 m, open / cubicle / wall-facing booth layouts, and desks with or without clutter;
- **seed robustness**: 12 independent six-week worlds for each of 4 layouts;
- **sensitivity**: 17 one-at-a-time changes to drone speed, climb speed, energy weight, time budget, observe time, detection range and altitude sets;
- a **look-alike stress test**: how many objects look alike, how well appearance matching works, and whether the policy uses its belief about where the object probably is.

**The altitude rule was fixed before the sweep ran.** Altitude stays a main claim only if some layout saves ≥ 10 % time by using all altitudes rather than 0.4 m only, with the 95 % interval above 0. If no layout reaches 3 %, the paper leads with the prior benchmark and look-alike re-identification instead.

| Question | Answer (Student-t 95 % intervals across seeds) |
|---|---|
| **Does altitude pay off?** | **No (rule: FAIL).** The best of 20 layouts saves 1.7 % [−0.5, 4.0]; none reaches 3 %. Flying *always* at 1.8 m instead of 0.4 m saves at most 3.3 % [0.8, 5.8]. No drone setting changes this. |
| Why not? | To see over an occluder, the drone must look down steeply from close by (see `test_seeing_over_an_occluder_needs_a_steep_angle`). Flying round it at 0.4 m costs about as much as climbing. In these spaces the remaining gap to an oracle is search and observation overhead, not sight lines. |
| Does the trust / verify / search policy help? | **Yes, robustly.** It is 5–7 % faster than fly-there-and-search in every layout (intervals exclude 0), **17–20 % faster when the object actually moved**, and 1.4–2.3 % faster than always-verify. Success is the same (differences ≤ 0.3 points). This holds under every sensitivity setting. |
| Does the prior matter? | Our activity-aware Weibull prior has the lowest Brier score in 83 % of seeds: 0.111, vs 0.114 for the persistence filter (difference −0.0026 [−0.0050, −0.0002]) and 0.146 for the commonsense stand-in. **Inside the policy the miscalibrated commonsense prior costs 3 points of success** in all four layouts, because it makes the drone accept a look-alike elsewhere. |
| Look-alikes | Every instance-level failure is a look-alike swap; intent success is 100 %. Using the belief to choose between look-alikes adds **+1 point (4 chairs) → +6 (12) → +10 (20 chairs) → +14 to +15 points (5 look-alike classes)**. The gain vanishes once look-alikes are distinguishable (appearance spread 0.06). |
| Language grounding (E3) | The staleness-aware resolver is **+2.7 to +2.8 points** more accurate than map-only, with intervals above 0. |

**Decision (point model; superseded by the Stage 3a update below):** under the pre-registered rule, **altitude is not a main claim**. The paper should lead with:
1. the prior-calibration benchmark (E1);
2. the cost-aware trust / verify / search policy, which is 17–20 % faster on moved objects;
3. look-alike re-identification using the belief as a spatial prior.

Altitude stays in as a reported negative result plus the fixed-height ablation. Stage 3 (Isaac Sim) should still include one scene where only an aerial view can see the target, to test the geometric explanation, rather than to rescue the claim.

## Stage 3a update: the altitude result reverses once visibility is calibrated on renders

The "no" above came from the point visibility model. Isaac Sim renders (3 layouts, 30 viewpoints × 3 altitudes × 4 yaws each; `../isaac/`) show that model under-counts what altitude reveals. The calibrated extended model (`+top`, a50 = 224.5 px, held-out agreement 96.2 %) gives a different answer, and the gain has a narrow cause.

| Run | Altitude saving (all 20 layouts) | Where it comes from |
|---|---|---|
| Point model (above) | ≤ 1.7 % — FAIL | — |
| Calibrated, laptops as closed 3 cm slabs (`results/stage2_calibrated/`) | ~63 % — PASS | laptops (success 0.15 → 1.00) and mugs (0.81 → 1.00) |
| Calibrated, laptops open, 25 cm tall (`results/stage2_calibrated_laptop_open/`) | 38–39 % — PASS | mugs only (0.81 → 1.00, mean 77 s → 16 s) |

Every other class shows no altitude effect in either calibrated run.

**The renders back up this mechanism and say the model is conservative.** Render-visible rate at 0.4 / 1.0 / 1.8 m:
- closed laptop: booth 1.3 / 6.0 / 10.7 %, open layout 2.0 / 10.7 / 26.7 %, cubicle 0.7 / 9.3 / 14.0 %;
- mug: booth 2.4 / 4.8 / 8.1 %, open layout 2.9 / 11.9 / 17.1 %, cubicle 1.9 / 4.8 / 9.0 %.

From 0.4 m the desk edge hides them. The model still says 7–19 % are visible at 0.4 m, so it *overstates* low-altitude visibility of small, flat objects, and the true gain is likely larger. Renders also show gains the model misses for monitors and chairs, e.g. 10.6 → 29.4 % and 16.7 → 39.4 % in the open layout.

**Revised claim.** "Altitude helps" is too broad. The defensible claim is that **a vantage point pays off for small or flat objects on surfaces above camera height, and is irrelevant for floor-standing ones**. The size of the gain therefore depends on what the user asks for. The paper should report it per object class, not as one number.

Open-laptop check (open layout, re-rendered with `export_scene.py --laptop-open`): open laptops are render-visible at 10.0 / 22.0 / 40.0 % from 0.4 / 1.0 / 1.8 m, against 2.0 / 10.7 / 26.7 % for closed ones. Opening the lid makes them visible from low down more often, but altitude still multiplies visibility by about 4×. The calibrated simulator with open laptops shows *no* laptop altitude effect, so it understates this case. The true laptop effect lies between the two simulated runs.

## Stage 3b update: a real detector and real appearance features

Isaac Sim renders with realistic library models, run through YOLO-World and DINOv2 (`../isaac/results/open_real_DETECTION.md`,
`open_real_REID.md`).

| Question | Answer (open layout, 360 frames) |
|---|---|
| How many pixels does a real detector need? | a50 ≈ 340 px with the right label, ≈ 270 px for a proposal under any label. The simulator assumed 224.5 px. The real curve is also shallower (slope 1.0 vs 0.35) and tops out at 85 %. |
| Does altitude survive a real detector? | **Yes.** Correctly labelled detections per viewpoint: 5.2 → 7.1 → 9.3 at 0.4 / 1.0 / 1.8 m, +79 %, vs +80 % visible in the renders. Mugs: 2.9 → 14.3 % detected. Closed laptops seen from above are the weak spot (≤ 5 %). |
| Are the detector's labels reliable? | No. Trash can → "cup" 63 %, stool → "chair" 77 %, monitor → "laptop" 35 %. Boxes are still found: ≥ 87 % of clearly visible objects for most classes. So detection gives proposals, and identity comes from re-identification. |
| Does metric size from depth help? | The nearest-size rule picks the true class in 73 % of the detector's confusions: stool / chair 86 %, monitor / laptop 93 %, bin / cup 100 %. It fails when the box covers only part of the object (a chair back called "monitor"). |
| Can appearance (DINOv2) tell objects apart? | Between classes, fairly well (AUC 0.88). Between same-class objects, barely (AUC 0.57; 0.59 with flat colour tints). In the simulator, only chairs were look-alikes. In the renders, nearly every class behaves like one: a same-class spread of 0.015 instead of 0.5. |

**Stage 2 re-run with the real detector's curve** (`results/stage2_realdet/`: a50 339.6 px, slope 1.0, p_max 0.85):

- **Altitude:** PASS in all 20 layouts, saving 46–48 % across seeds. As before, the saving comes from laptops (success 0.43 → 1.00) and mugs (mean time 42 → 19 s). The simulator uses one detection curve for all classes, so it overstates how detectable closed laptops are from above, and the laptop part of the gain is optimistic.
- **Policy vs trust:** 10–12.5 % faster (moved objects: 9–17 %), and 2–3 % faster than always verifying. Success is 0.3–0.6 points lower than trust, with an interval excluding 0 in two layouts. With a less reliable detector, verifying from far away sometimes misses an object that flying close would find.
- **Priors:** the commonsense stand-in prior still costs about 2.8 points of success.

## Limits of this simulation

- Detection depends only on line of sight and range. No lighting, motion blur or detector false positives.
- Re-ID uses synthetic embeddings.
- Humans are not simulated as moving obstacles, and the safety filter is not modelled.
- The drone is a kinematic point with constant speeds.
- Object movement is a stationary renewal process, tuned by hand to be plausible. Real logs will differ, which is the point of E1.
