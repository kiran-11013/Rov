# The Half-Life of VLM Decisions in Robot Controllers

**One line:** VLM-driven robots must decide when to re-ask their VLM. We measure how long VLM-set control decisions actually stay valid, why they change, what acting on a stale one costs physically, and how well each kind of re-query trigger catches the changes that matter.

**Status:** v5, the canonical proposal. Consolidates four rounds of hostile review (history in `README.md`).
**Author:** M.Tech, IIT Hyderabad · **Target:** IROS 2027 (primary), RA-L (alternative)
**Platform:** 1–2 kg PX4 quadrotor, ZED stereo, mono cameras, LiDAR, IMU, AprilTags, RTX 4090, Jetson Orin

---

## Abstract
Vision-language models (VLMs) increasingly set parameters inside robot control loops: safety margins, planner weights, impedance gains. Because VLM calls are slow, every such system embeds an assumption about how often the VLM must be re-queried. Fixed rates, scene-change triggers, embedding gates and dual-system architectures are all in use, but the rate and causes of VLM decision change have never been characterised for robot controllers. We record ≈ 300 drone-height RGB-D episodes and replay two VLM-driven controllers with two independent VLM families on every frame. We estimate the **survival curve ("half-life")** of VLM decisions and attribute each change to one of four causes: geometric change, semantic change, sampling noise and perceptual jitter. We use a factorial image × geometry design for this. The physical cost of staleness is measured through branching closed-loop rollouts with a latency-and-queue model, calibrated on 72 real flights. We then rank trigger families at matched call rates by harmful-change recall and energy. The result tells practitioners how much of their VLM compute buys safety, how much buys noise, and where learned gates are worth paying for.

## 1. Origin (why this line of work)
This started as an audit of **ImpedanceGPT** (IROS 2025, arXiv 2503.02723). In that paper a VLM + RAG system picks impedance parameters for a drone swarm. The audit found five problems:
1. The VLM runs **once, offline**, taking 5–8 s, then picks 1 of 20 scenarios.
2. The 20 scenarios collapse to roughly two parameter sets, so retrieval reduces to a hard/soft switch.
3. In the released code, the obstacle type is passed via the **filename**, not inferred by the VLM.
4. There are no baselines, no ablations, ~21 trials, and only peak speed is reported as the result.
5. The core open question is never asked: **when, and how often, does a VLM-in-the-loop controller need to ask its VLM again?**

## 2. Problem and gap
- **In use today:** VLM-set control parameters (AlphaAdj: VLM → CBF conservativeness; EAMP: VLM → planner parameters). Re-query policies range from fixed rate to event triggers (PC-SET, React-When-You-Need-To), embedding gates (AESOP), latency-tolerant designs (Slow Brain, Fast Planner) and adaptive invocation (ProAct-VLM).
- **Already known (video analytics):** run a reference model on every frame and learn cheap filters against it (NoScope, Reducto, Chameleon, FilterForward).
- **Gap:** no one has measured, **for robot controllers**:
  - (a) how long VLM-set parameters stay behaviourally valid;
  - (b) what fraction of changes are noise rather than real change;
  - (c) what staleness costs **physically** in closed loop, including VLM latency;
  - (d) how trigger families compare at matched call rates, separately on geometry-visible and semantic-only events.

**What differs from video analytics:**
1. The "query result" is a **control parameter**, judged by **behavioural** equivalence, not label equality.
2. The cost is **physical** (clearance, intrusion), not query accuracy.
3. The reference model itself **is noisy** (sampling, jitter), and that noise is a measured cause.
4. Semantic changes can have **small pixel deltas** (person ↔ mannequin), which is exactly where frame differencing should fail.

## 3. Research questions
| | Question | Primary output |
|---|---|---|
| **RQ1 Half-life** | How long do VLM-set decisions stay valid in natural footage, and what share of changes come from geometry, semantics, sampling noise and perceptual jitter (plus interaction)? | Kaplan–Meier survival curves per host × VLM, split by cause |
| **RQ2 Consequence** | What does acting on a stale decision cost in closed loop, given realistic VLM latency, and does a well-designed host absorb it? | Δ min clearance and proxemic-intrusion curves (0.6–2.0 m sweep); efficiency cost (path time, energy) |
| **RQ3 Triggers** | At matched call rates, which trigger families catch the harmful changes, on geometry-visible vs semantic-only events, and at what energy? | Recall vs call-rate vs joules Pareto; **one pre-registered primary test** |

**Target headline (must survive any outcome):**
> "X % of VLM decision flips are noise; Y % are behaviourally irrelevant; of the Z % that matter, a geometric heuristic catches W % at 0.5 Hz. The semantic-only residual is where learned gates earn their compute."

## 4. Method
### 4.1 Data
| Set | Size | Use |
|---|---|---|
| **Natural** | ≈ 240 handheld ZED episodes at drone height, 60 s each, 2–3 indoor environments (corridor, lab, atrium), people going about normal activity | RQ1 rates, RQ3 main ranking |
| **Staged** | ≈ 60 episodes of counterfactual pairs: same geometry with semantics swapped (person ↔ mannequin, sign added, person picks up a ladder) and vice versa | Attribution validation, RQ3 stress test, the semantic-only stratum |
| **Flight** | ≈ 72 paired runs (12 scripted intrusions × 3 arms × 2 repetitions), mannequin on a cart, mocap/AprilTag ground truth | Validating the replay/rollout model |

Decision rate 15 Hz → ≈ 270 k frames. Natural and staged data are **never pooled** for rates.

### 4.2 Hosts (two VLM-in-flight controllers)
1. **VLM-regulated CBF safety margin**, AlphaAdj-style, including its stale-request mitigation, built on the open ASMA safety-filter code where possible.
2. **VLM planner-parameter retuning**, EAMP-style. Its own PC-SET trigger is included as a benchmark arm.

Published prompts are used verbatim where available, with **3 prompt paraphrases per host**. If variance across paraphrases exceeds variance across hosts, that is reported as a finding.

### 4.3 VLMs
Two genuinely independent families, e.g. **Qwen2.5-VL-7B** and **InternVL3 / Gemma-3**. Run at the deployed temperature (T = 0 where hosts use it). Sampling noise is measured at each host's published temperature.

### 4.4 Dense replay with latency
- Every host × VLM × paraphrase runs on **every frame** offline (RTX 4090; ≈ 80–150 GPU-hours, to be measured in the pilot).
- A **latency + queue model**, using latency distributions measured on Jetson Orin, turns oracle decisions into what each trigger would actually have had in hand. Staleness = trigger delay + queue + inference.
- **Pipelined always-call** is the zero-trigger baseline.
- Wording: "open-loop-exact" for decision sequences; closed-loop effects come only from 4.6.

### 4.5 Behavioural equivalence and cause attribution
- **Equivalence:** decisions d, d′ are equivalent if the host controller's commanded velocity under them differs by ≤ ε = 0.1 m/s over H = 2 s (sensitivity analysis over ε, H).
- **2×2 factorial attribution:** for each non-equivalent change t → t′, query the VLM on (image_t or image_t′) × (structured geometry_t or geometry_t′). Shapley values give **geometric**, **semantic** and **interaction** shares.
  - **Sampling noise:** resample the same input.
  - **Perceptual jitter:** consecutive static frames.
  - Cost ≈ 32 k extra calls.

### 4.6 Physical consequence: branching rollouts
- At each decision change, reset to the logged state and roll out a **point-mass / PX4-SITL** drone model for H ≤ 3 s under the stale vs the fresh decision.
- Rollouts run in a ZED-reconstructed static scene with replayed human tracks.
- **Outputs:** Δ minimum clearance; **proxemic intrusion** curves over thresholds 0.6–2.0 m (sourced numbers, e.g. 1.84 m mean preferred distance, Wögerbauer et al. 2024); efficiency cost.
- Every harm metric is stratified into **geometry-visible** and **semantic-only** events.
- The rollout model is calibrated on the 72 flights (predicted vs measured clearance: R², bias).
- **Limitation:** replayed humans are non-reactive.

### 4.7 Trigger benchmark (RQ3)
| Family | Arm |
|---|---|
| Fixed rate | every k frames, randomised phase |
| Geometric heuristic | new/lost track, label change, Δdistance / Δvelocity > θ |
| Image-feature change | vision-encoder feature distance (React-When-You-Need-To style) |
| Semantic event trigger | PC-SET (EAMP) |
| Text-embedding gate | e5/MiniLM + LR; AESOP-style kNN |
| Small local LLM | Qwen 0.5–3B yes/no logprob, temperature-scaled |

**Protocol**
- Sweep each trigger's threshold; interpolate recall at fixed rates {0.1, 0.25, 0.5, 1, 2} Hz; report recall-AUC over log(rate) and the per-episode p95 rate (burstiness).
- Energy per frame (Jetson INA rails) **including the always-on perception**.

### 4.8 Statistics
- **Unit of analysis:** episode; episode-cluster bootstrap CIs everywhere.
- **Primary test (pre-registered):** best non-heuristic trigger vs geometric heuristic, harmful-change recall at 0.5 Hz, natural data, paired.
- **Everything else:** Holm-corrected, reported as secondary.
- **Power note:** ~100 natural harmful events give ±10 pp on recall, so only differences ≥ 15 pp are claimed.

## 5. Typed calibrated gates (Jev): secondary study
The original motivation included Jev (TypeSafe AI, Sept 2026), a hosted, text-only model that returns typed answers with calibrated probabilities.
- **Role:** an **appendix study** and a **separate workshop paper**, not in the main reproducible ranking.
- **Protocol:** Jev answers "is the last decision still valid?" on the structured JSON state, called **only on frames the geometric heuristic flags as quiet** (this respects rate limits and targets the semantic-only residual).
- **Logging:** version pinned, every request and response logged.
- **Workshop paper:** latency/RTT from the flying drone, timeout rate, calibration (ECE) on embodied Booleans before and after recalibration.

## 6. Contributions
1. **Decision half-life:** a cause-attributed, consequence-weighted measurement of how long VLM-set control parameters stay valid, across 2 hosts × 2 VLM families × 3 prompts.
2. **A measurement protocol** that extends video-analytics dense replay to control: behavioural equivalence, factorial cause attribution, latency/queue modelling, and branching rollouts calibrated on real flights.
3. **A matched-rate ranking** of re-query trigger families on geometry-visible vs semantic-only harm, with energy.
4. **The open "Expiry" dataset and benchmark** (natural + staged + flight).

## 7. Plan (9 months)
| Month | Work | Gate |
|---|---|---|
| 1 | Host 1 + 30-episode pilot + Orin latency profiling | **Go/no-go** (§8) |
| 2 | Host 2, rollout model, staged-pair protocol | |
| 2–4 | Recording (natural + staged) | |
| 4 | Dense replay, attribution, rollouts | |
| 5–6 | 72 flight runs, calibration of the rollout model | |
| 6–7 | Trigger benchmark, statistics; Jev appendix + workshop draft | |
| 8–9 | Writing, dataset release, submission | |

## 8. Go/no-go (end of month 1, numeric)
| Check | Pass |
|---|---|
| Non-equivalent decision changes in natural footage | ≥ 0.5 per minute |
| Sampling-noise flips | < ⅓ of that rate |
| Harmful changes missed by the geometric heuristic at 0.5 Hz | ≥ 15 % → the learned-gate section stays; otherwise the paper reports "heuristics suffice" (still publishable) |

## 9. Risks
| Risk | Mitigation |
|---|---|
| Host 1 harm ≈ 0 (CBF + stale cap absorb it) | Pre-registered as an acceptable finding; report the efficiency cost instead |
| Decision changes rare | Staged stratum guarantees coverage; natural rate is reported as is |
| Noise dominates | Headline finding ("most VLM flips are noise") motivates temporal smoothing |
| Re-implementations unrepresentative | Verbatim prompts, open code, 3 paraphrases, paraphrase-variance reported |
| Handheld ≠ flight | Rollouts calibrated on flights; distribution shift quantified |
| Workload for one student | Month-1 scope is one host; compute is not the bottleneck, person-months are |
| No Jev access | Jev is appendix/workshop only; the main paper is unaffected |

## 10. Audit trail (how the idea got here)
| Version | Idea | What killed it |
|---|---|---|
| — | Audit of ImpedanceGPT | Motivated the line of work (§1) |
| Two Witnesses | Depth + VLM cross-check with conformal safety | Wrong direction of harm; empty guarantee; strawman attacks; prior art (AlphaAdj, ASMA, semantic CBFs) |
| See the wind | Semantic aerodynamic memory | Dropped by author choice (goal = compute efficiency with Jev) |
| v1 REFLEX | Jev replaces VLM decision + certified escalation | A decision tree matches Jev; gate blind to perception errors; ≥ 5× impossible vs short-JSON prompting |
| v2 Saccade | Jev schedules VLM calls | Same Jev-vs-classifier problem; certified cascades exist |
| v3 Selective reuse | Certified reuse of VLM decisions | False novelty (AESOP, React-When-You-Need-To, EAMP); circular harm; empty certificate |
| v4 Expiry | Measurement + trigger benchmark | Open-loop counterfactual invalid; latency ignored; NoScope/Reducto prior art; weak statistics |
| **v5 (this)** | **Decision half-life with causal attribution, closed-loop rollouts, matched-rate trigger ranking** | Estimated IROS 45–50 % if executed as written (reviewer estimate) |

**Where we disagree with the last audit:**
1. The reviewer wanted "≥ 15 % missed by heuristic" as a hard go/no-go. We use it only to gate the learned-gate section, because the paper's headline survives either outcome.
2. Jev stays as a pre-registered appendix study rather than being dropped, to meet the author's goal without contaminating the main ranking.

## 11. References (verify every entry; several are from search snippets)
ImpedanceGPT arXiv 2503.02723 · AlphaAdj arXiv 2603.21142 · ASMA arXiv 2409.10283 (code: github.com/souravsanyal06/ASMA) · EAMP/PC-SET arXiv 2606.25629 · React When You Need To arXiv 2609.22587 · AESOP (Sinha et al., RSS 2024) · Slow Brain, Fast Planner arXiv 2606.20458 · ProAct-VLM arXiv 2609.37681 · Act-Think-Abstain arXiv 2603.05147 · VLA temporal redundancy arXiv 2607.12287 · VLA-Cache arXiv 2502.02175 · Hi Robot arXiv 2502.19417 · Fast-in-Slow arXiv 2506.01953 · NoScope (VLDB 2017) · Reducto (SIGCOMM 2020) · Chameleon (SIGCOMM 2018) · FilterForward (MLSys 2019) · Wögerbauer et al. 2024, preferred distance in human–drone interaction (PMC11503297) · TypeSafe Jev announcement (typesafe.ai, Sept 2026)
