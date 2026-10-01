# Saccade — Calibrated, Certified Scheduling of VLM Calls for Compute-Efficient Semantic Drone Control

**Status:** v2. Reframed after the hostile audit of REFLEX v1 (`reflex_proposal_v1.md`).
**Non-negotiable goal kept:** show that integrating Jev (TypeSafe's System-One decision model) into VLM-driven robot pipelines cuts compute while keeping accuracy.

---

## 0. Why v1 was reframed (audit summary)

| v1 flaw | Consequence | v2 fix |
|---|---|---|
| Jev decided from JSON that code had already resolved | A 0.1 ms decision tree matches it | The state carries **open-vocabulary text**; Jev's advantage is tested **zero-shot on unseen object types** |
| The gate measured decision ambiguity, not perception error | High-confidence wrong answers | The gate asks whether the **last decision still holds given scene changes**, using perception-disagreement features |
| Most of the saving came from shorter decoding (A6 gets it without Jev); the rest is prefill | ≥5× impossible; p95 worse than A0 | The saving is **VLM calls skipped entirely**, removing prefill and decode |
| Guarantee on 150 frames from ~40 episodes | Bound looser than ε | Episode-level Clopper–Pearson on ≥150 handheld-rig episodes + an in-flight validation split |
| ImpedanceGPT is a strawman host (offline, near-binary) | No credit for beating it | Main host is a pipeline where the VLM runs **in flight**; ImpedanceGPT is secondary |
| Static scenes: speed buys nothing | RQ4 vacuous | Dynamic hazards with a crossover analysis |
| False novelty claims | Desk reject | Positioned against certified cascades (see §6) |

## 1. Problem

VLM-driven drone controllers call a 7B VLM to re-derive a semantic decision (a safety margin, impedance parameters, a waypoint). On a Jetson Orin, each call costs 1–6 s, almost all of it image prefill plus decode. Most consecutive frames do not change the decision. They either call the VLM continuously (too slow and power-hungry) or once per scene (blind to change).

## 2. Research question

> **Can a calibrated, typed decision model decide *when* a drone needs to call its VLM, skipping most calls with a certified bound on harmful stale decisions, and does this generalise to object types the scheduler was never trained on?**

| ID | Hypothesis | Pass criterion |
|---|---|---|
| **H1 (make-or-break)** | On a JSON state with open-vocabulary object text, Jev beats a trained router (LR / decision tree / kNN on text embeddings) on **held-out object categories** | ≥ +3 pts accuracy or ≥ 20 % lower AURC, 95 % episode-bootstrap CI excluding 0 |
| H2 | Saccade skips the VLM on ≥ 70 % of frames | Cost-weighted harmful-staleness rate ≤ certified bound (Clopper–Pearson UCB, δ = 0.05) |
| H3 | Skipping calls buys behaviour in dynamic scenes | Success-vs-hazard-onset-distance curve shows a crossover where always-VLM (latency) and blind reuse (staleness) fail and Saccade succeeds |
| H4 | Honest compute accounting | On-device latency, energy (INA rails), off-device estimate, $/decision and RTT from the flying drone reported separately; fully local arm is the main result |

## 3. Method

```
every frame (cheap, on-device):
  open-vocab detector + ZED depth + tracker  →  JSON state
     {objects:[{text:"person pushing a cart", dist_m, bearing, vel, track_age}],
      delta_since_last_vlm:{new_tracks, lost_tracks, label_changes, max_dist_change},
      perception_flags:{detector_vs_depth_disagree, low_light}}
  Jev (typed, one pass):
     Boolean  still_valid   — does the last VLM decision still hold?
     Boolean  unaccounted   — is anything present the last decision ignored?
  if P(still_valid) ≥ τ* and P(unaccounted) ≤ τ'*: reuse last decision
  else: call VLM (host pipeline) → new decision, reset reference state
fast safety layer (100+ Hz, always on): depth-based CBF with conservative default margin
```

- **Arithmetic stays in code** (distances, velocities, deltas). Jev only answers semantic Booleans.
- **Threshold selection:** LTT with Clopper–Pearson on episode blocks. Cost-weighted harm: "hard/aggressive decision kept while a human is within X m" is weighted ≫ "an unnecessary VLM call".
- **Model-agnostic:** the decision layer can be Jev (hosted), an open RLCD-style reproduction (e.g. qwen-rlcd), or open-jev. The **fully local arm is the headline**; Jev is one instantiation. All Jev requests and responses are logged and the model version pinned.

## 4. Hosts

1. **Primary:** a VLM-in-flight pipeline. Either a VLM-tuned safety margin (ASMA/AlphaAdj-style CBF) or See-Point-Fly-style pointing → waypoint.
2. **Secondary:** ImpedanceGPT (single drone, egocentric ZED), scored on **parameter-equivalence classes**, not its 20 nominal scenarios.

## 5. Experiments

**Data**
- **Calibration:** ≥ 150 episodes from a handheld ZED rig at drone height (cheap, no flight risk).
- **Validation:** ≥ 40 in-flight episodes held out to check that the bound transfers.
- **Unseen categories:** reserved for H1 (e.g. train on person/stand/chair; test on cardboard cutout, child on scooter, wet-floor sign, ladder).

**Arms**

| Arm | Description |
|---|---|
| S0 | Always-VLM (host pipeline every frame / at max rate) |
| S1 | Once-per-scene VLM (blind reuse) |
| S2 | Fixed-period VLM (every k frames) |
| S3 | Heuristic trigger (new/lost track or distance change > θ) |
| S4 | Trained router (LR/tree/kNN) on the same JSON |
| **S5** | **Saccade with Jev** |
| **S6** | **Saccade with an open local decision model** |
| A6 | Short-JSON-prompted VLM every frame (the "just prompt better" control) |

**Metrics:** VLM-call rate; harmful-staleness rate with UCB; decision accuracy on parameter-equivalence classes; latency (median/p95, on-device vs network); Jetson energy; $/decision; AURC; closed-loop success and minimum clearance vs hazard-onset distance.

**Profiling first:** prefill vs decode for Molmo-7B (4-bit) and one 2–3B VLM on the RTX 4090 and on Orin, same quantization.

## 6. Positioning (honest)

- **Certified cascades / routing already exist:** Conformal Cascade, Trust-or-Escalate, Calibrate-Then-Delegate, C3PO, FST cascades, FrugalGPT/RouteLLM. Saccade differs in three ways:
  1. **Temporal reuse:** it schedules *when to re-perceive* in a stream, rather than routing independent queries.
  2. **Embodied, cost-weighted harm** with perception-dominated errors.
  3. **Zero-shot generalisation of the scheduler to unseen object semantics**, tested against trained routers.
- **Robotics efficiency work:** See-Point-Fly, LiteVLA-H (dual-rate on Orin), VLA-AN. Saccade is complementary: it reduces *how often* any of them is called.

## 7. Make-or-break plan (2 weeks)

1. Profile prefill/decode on 4090 + Orin.
2. Collect ~300 labelled frames from ≥ 40 handheld episodes, including held-out object categories.
3. Run H1: Jev (or open-jev if no API key by day 5) vs LR/tree/kNN on the same JSON, 5-fold CV by episode.
4. **Decision rule:**
   - H1 passes → build Saccade.
   - H1 fails → pivot to a multi-host study, "Do VLM controllers need generative decoding?", with Jev reported as a negative result.

## 8. Risks

| Risk | Mitigation |
|---|---|
| No Jev access | Open-model arm is the headline; Jev is optional |
| Jev ties with LR | Pre-registered pivot (§7) |
| Handheld → flight distribution shift breaks the bound | Separate in-flight validation; report the bound's empirical violation rate |
| Open-vocab detector misses the hazard entirely | The always-on depth CBF is the safety floor; Saccade only governs semantic refinement |
