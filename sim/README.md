# Look Before You Fly — simulation of the complete algorithm

This folder implements every algorithmic part of `research/PROPOSAL.md` (v7) in simulation, before any hardware work:
six weeks of object movement, the persistence priors (E1), the trust / verify / search policy with aerial
viewpoints, re-identification, and all E2 arms including the fixed-height ablation, plus grounding of
language commands with conformal clarification (E3).

```bash
pip install -r requirements.txt
python run_all.py            # both layouts, about 40 s → results/{open,cubicle}/RESULTS.md + figures
python run_all.py --quick    # smoke run
python -m pytest             # 11 tests
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

## Limits of this simulation

- Detection depends only on line of sight and range. No lighting, motion blur or detector false positives.
- Re-ID uses synthetic embeddings.
- Humans are not simulated as moving obstacles, and the safety filter is not modelled.
- The drone is a kinematic point with constant speeds.
- Object movement is a stationary renewal process, tuned by hand to be plausible. Real logs will differ, which is the point of E1.
