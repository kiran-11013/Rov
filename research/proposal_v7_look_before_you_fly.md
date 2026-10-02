# Look Before You Fly: Acting on Stale Object Memory with an Aerial Vantage Point

**One line:** A drone maps a space and gives every object an ID. Later you say *"go to #7"* or *"go to the chair by the door"*. By then #7 may have moved. Should the drone trust its map, **look from altitude first**, or search? We answer this with six weeks of tracked ground truth and real flights.

**Status:** v7, the canonical proposal. Applies the audit of v6 (`audit_v6.md`). History in `README.md`.
**Author:** M.Tech, IIT Hyderabad · **Target:** IROS 2027 (primary), RA-L (alternative)
**Platform:** 1–2 kg PX4 quadrotor, ZED stereo + Jetson Orin (positional tracking, spatial mapping, object tracking), mono cameras, AprilTags for ground truth, RTX 4090 offboard. LiDAR is used only if the ZED fails.

---

## Abstract
Robots that take commands like "go to #7" act on object memories that silently expire: objects get moved between mapping and command. Persistence models (the persistence filter, temporal persistence modelling, PreSIST) estimate how likely a remembered object is still in place. Systems such as Bogenberger et al. revisit stale regions. **What is missing is the command-time decision for a specific remembered instance:** fly straight there, verify first from a cheap vantage point, or search. Also missing is how a drone's ability to **look from altitude** changes that decision.

We contribute:
1. A **six-week tracked-ground-truth benchmark** (≥ 60 objects, ≥ 10 classes) that tests existing persistence priors for command-time use.
2. A **cost-aware trust / verify / search policy** in which verify chooses viewpoint *and altitude* to maximise the probability of observing the object per unit of time and energy.
3. **Language grounding under stale memory**, with calibrated clarification.

A fixed-height ablation of the same drone isolates the value of altitude.

## 1. Origin
This line of work began with an audit of **ImpedanceGPT** (IROS 2025, arXiv 2503.02723):
- The VLM ran once offline and picked 1 of 20 near-duplicate scenarios.
- The obstacle type was passed via the filename.
- There were no baselines.

Six hostile reviews later (audit trail, §12), three lessons hold:
1. Claim only what prior work does not already hold.
2. Measure success against **physical ground truth**.
3. Pre-register outcomes that are still publishable if the main hypothesis fails.

## 2. Related work and gap
| Work | What it does | What remains for us |
|---|---|---|
| Persistence filter (ICRA'16); TPM (Toris & Chernova, ICRA'17); Bore et al. (T-RO'19); FreMEn | Learned survival / periodic presence of objects; search when others move objects | Aerial vantage; instance commands; action choice |
| **PreSIST** (2607.04057) | VLM → per-object survival prior (Weibull) fused with a persistence filter | Used here as a **baseline prior**, evaluated inside an action policy |
| Perpetua (IROS'25); PredictiveGraphs (2605.00121); FlowMaps (2606.20209); Patel & Chernova (CoRL'22) | Multimodal persistence, cyclic states, where moved objects go | Used as priors for search |
| **Bogenberger et al.** (RA-L'26) | Per-instance stationarity, active revisits, LLM object-goal navigation in real homes | **Closest system.** No specific-ID commands, no trust/verify/search choice, ground robot |
| LT-Mem (2608.19059); VLMM (2607.16173); Memory-for-Attention (2607.23797) | Volatility-aware memory; which objects to re-observe under a budget (simulation) | Physical action cost in real flight |
| Quadrotor InstanceImageNav (2606.29917) | Drone flies to a specific instance with viewpoint planning | Stale memory; language |
| KnowNo (CoRL'23); ESC / L3MVN | Conformal ask-for-help; LLM search priors | Baselines |

**Gap (one sentence):** no work studies the **command-time choice between trusting, verifying and searching** for a **specific remembered instance**, when the robot can **verify cheaply from altitude** and the **language reference itself may be stale**.

## 3. Research questions
| | Question | Primary output |
|---|---|---|
| **RQ1 Priors** | How well do existing persistence priors predict whether a specific remembered object is still within 1 m of its mapped pose, at command time, in a real in-use space? | Time-dependent Brier score / log loss over object-time pairs; restricted mean survival time (RMST) at 1 h and 1 day per class |
| **RQ2 Action** | Does a cost-aware trust / verify / search policy with aerial verify beat trust-then-search and verify-always? How much does altitude contribute? | Success on true current ground truth; time and energy to success; fixed-height ablation |
| **RQ3 Grounding** | Can language commands be resolved to IDs *jointly with* persistence ("the chair by the door" now vs yesterday), with calibrated clarification? | Accuracy, risk–coverage with CIs, clarification rate |

**Headline number:**
> "After one day, a map-trusting drone reaches the right object X % of the time; looking from altitude first raises this to Y % at a cost of Z seconds."

**Pre-registered alternative outcome:** if verify-always is optimal, the paper reports **"altitude makes memory cheap to check"** and leads with the benchmark and the prior-calibration result.

## 4. System
1. **Mapping (ZED + Orin).**
   - Positional tracking and spatial mapping build the metric map.
   - An open-vocabulary detector proposes objects; depth lifts them to 3D; instances are associated across views.
   - Each object gets an ID, class, VLM-written description, pose, covariance and `last_seen`.
2. **Persistence belief.** P(within 1 m of mapped pose | class, Δt, time of day), from a hierarchical Weibull / piecewise hazard fitted on E1, with a return probability for periodic objects. Alternative priors are plugged in for comparison (§5).
3. **Where it went.** A displacement distribution (FlowMaps-style, from E1) plus an ESC-style LLM commonsense prior for the search phase.
4. **Policy.** Choose the action with minimum expected cost, where cost = time + λ·energy:
   - **Trust-then-search:** fly to the mapped pose; search if absent.
   - **Verify:** pick the viewpoint *and altitude* that maximise P(observe | belief) per unit cost from a vantage graph; observe; then commit or search.
   - **Search:** belief-weighted coverage.
5. **Re-identification.** Appearance embedding + VLM description + spatial prior.
   - Success is reported both **instance-exact** and **intent-equivalent** (any object satisfying the reference).
   - Identical objects are reported as a separate subset.
6. **Grounding.** P(ID | utterance, map, Δt, persistence). The drone asks a clarifying question when the conformal prediction set has more than one ID.
7. **Safety floor.** An always-on depth CBF keeps clearance from obstacles and people.

## 5. Experiments
### E1: six-week persistence benchmark (RQ1)
- **Scope:** ≥ 60 tagged objects in ≥ 10 classes (chairs, stools, carts, bags, boxes, bins, laptops, monitors, plants, toolboxes…), across **≥ 3 connected spaces** (lab, corridor, adjacent room/store), for ≥ 6 weeks.
- **Ground truth:**
  - AprilTags on top surfaces, read by fixed external cameras.
  - Tag pixels are **masked in drone frames**, with a masking ablation.
  - ≥ 100 manual audits report the ground-truth error.
- **Event definition:** displacement > 1 m, or absence. Time of day is modelled; return probability is reported.
- **Analysis:**
  - A class is reported only with ≥ 15 events; rarer classes are pooled hierarchically.
  - Priors compared: persistence filter, TPM-style fit, PreSIST-Lang, a constant, **10 human guessers**, small local LLM, Jev.
  - Scoring: time-dependent Brier score over object-time pairs, with object-cluster bootstrap.

### E2: command trials (RQ2)
- **Main table:** counterfactual replay of every arm on E1 logs, using action-time and energy distributions measured in flight. The replay is validated against real flights (bias, R²).
- **Real flights:** ≥ 100 paired commands for A1 / A2 / A5 in the **1 h and 1 day** cells, across ≥ 3 spaces with occluders and targets at multiple heights (floor, table, shelf).

| Arm | Description |
|---|---|
| A1 | Trust-then-search |
| A2 | Verify-always |
| A3 | Persistence filter prior + our policy |
| A4 | PreSIST prior + our policy |
| A5 | **Ours** (fitted prior + cost-aware aerial verify) |
| A5-fixed | Ours at a fixed 0.4 m height (ground-robot equivalent) |
| A6 | Oracle (true current pose) |

**Metrics:** success (instance-exact and intent-equivalent), time and energy to success, wasted trips, ID errors, minimum clearance to people.
**Primary test (pre-registered):** A5 vs the better of A1/A2 on success within the time budget, paired, real flights. Holm correction for everything else.

### E3: grounding under stale memory (RQ3)
- **Commands:** ≥ 3 non-author command writers; ambiguity labelled before system output is seen; relational references whose answer changes with object movement.
- **Resolvers:**
  - KnowNo-style conformal sets over local-LLM logprobs;
  - embedding similarity;
  - Jev, a typed Choice over IDs taking per-ID persistence as input.
- **Metrics:** accuracy, risk–coverage with CIs, clarification rate.
- **Scale:** about 50 commands in flight, the rest offline.

## 6. Role of Jev
Jev (TypeSafe AI, Sept 2026) is hosted, text-only, typed, with calibrated probabilities. It has two roles:
- (a) **staleness-aware grounding:** a typed Choice over IDs with persistence probabilities as input;
- (b) one arm of the prior comparison in E1.

**Rules:**
- Version pinned; all I/O logged.
- Kept out of the primary test.
- **Hard drop date: end of month 2** if access is not granted. A local LLM fills both roles.

## 7. Phase 2: moving targets (future work)
"Follow #12": track with ZED object/body tracking and Kalman prediction, intercept at a CBF-enforced distance, re-ID after occlusion. Not part of the first paper.

## 8. Contributions
1. **A six-week tracked-ground-truth benchmark** testing existing persistence priors (including LLM-based and human priors) for command-time use, released with maps and commands.
2. **A cost-aware trust / verify / search policy** with aerial verify (viewpoint + altitude per joule), plus a fixed-height ablation that answers "why a drone".
3. **Language grounding under stale memory**, with conformal clarification, evaluated on non-author commands.
4. **Real-flight evaluation** across connected spaces, with replay validated against flights.

## 9. Plan (9 months)
| Month | Work |
|---|---|
| 0–1 | **Ethics application (IITH)**; tag objects; install external cameras; start E1 logging |
| 1–3 | Mapping, IDs, re-ID on drone (logging continues); Jev decision by end of month 2 |
| 3–4 | E1 analysis; policy + replay; E3 offline |
| 5–6 | Real flights (arms chosen by replay results) |
| 7 | Analysis; Phase 2 sketch |
| 8–9 | Writing, dataset release, submission |

## 10. Go/no-go (end of month 1)
| Check | Pass |
|---|---|
| Objects move | ≥ 15 events projected per class for ≥ 5 classes over 6 weeks |
| Mapping + IDs | ≥ 80 % of tagged objects get a correct, stable ID in one mapping pass |
| Re-ID | ≥ 70 % instance-exact on a scripted test (non-identical objects) |
| Ethics | Approval filed (granted by month 2) |

## 11. Risks
| Risk | Mitigation |
|---|---|
| Verify-always is optimal | Pre-registered outcome: "altitude makes memory cheap to check" |
| Few relocation events | More objects, a busier space, hierarchical pooling; natural and scripted reported separately |
| Ethics delay | File in month 0; logging can start with people-free camera views |
| Identical objects | Intent-equivalent success; identical subset reported |
| Ground truth leaks into perception | Tag masking + ablation |
| No Jev | Drop at end of month 2; local LLM |

## 12. Audit trail
| Version | Idea | Outcome |
|---|---|---|
| Two Witnesses | Depth + VLM cross-check | Wrong harm direction; prior art |
| v1 REFLEX | Jev replaces the VLM decision | Decision tree matched Jev |
| v2 Saccade | Jev schedules VLM calls | Certified cascades exist |
| v3 Selective reuse | Certified reuse of VLM decisions | False novelty; circular harm |
| v4 Expiry | Decision-staleness measurement | Open-loop replay invalid |
| v5 Half-life | Decision half-life study | Est. IROS 45–50 % if executed; replaced for a task-driven design |
| v6 "Is #7 still there?" | Object memory + navigation | False "first" claims (TPM, PreSIST); policy collapses in one room; drone unjustified. IROS 15–20 % |
| **v7 (this)** | **Aerial verify on stale object memory** | Reviewer estimate IROS 40–45 % if executed |

## 13. References (verify all; several from search snippets)
ImpedanceGPT 2503.02723 · PreSIST 2607.04057 · Persistence filter (Rosen et al., ICRA 2016) · TPM (Toris & Chernova, ICRA 2017) · Bore et al. T-RO 2019 (1801.09292) · FreMEn (Krajník) · Perpetua 2507.18808 · PredictiveGraphs 2605.00121 · FlowMaps 2606.20209 · Patel & Chernova CoRL 2022 · Bogenberger et al. RA-L 2026 (2509.19851) · LT-Mem 2608.19059 · VLMM 2607.16173 · Memory for Attention 2607.23797 · Quadrotor InstanceImageNav 2606.29917 · KnowNo CoRL 2023 · ESC 2301.13166 · Khronos · DovSG 2410.11989 · UAV-ON 2508.00288 · TypeSafe Jev (Sept 2026)
