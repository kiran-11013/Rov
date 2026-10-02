# Where Is #7 Now? A Drone Acting on Stale Object Memory

**One line:** A drone maps a space and gives every object an ID. Later you say *"go to #7"* or *"go to the mug by the window"*. By then #7 may have moved, and five other objects may look exactly like it. The drone must decide whether to trust its map, check first, or search. It must also recognise *which* look-alike is #7. We build this decision, test it in simulation and in Isaac Sim with a real detector, and then fly it.

**Status:** v8, the canonical proposal. It replaces v7 (`proposal_v7_look_before_you_fly.md`) after simulation and Isaac Sim evidence changed which claims hold (§6). History is in `README.md`.
**Author:** M.Tech, IIT Hyderabad · **Target:** IROS 2027 (primary), RA-L (alternative)
**Platform:**
- 1–2 kg PX4 quadrotor with ZED stereo + Jetson Orin, mono cameras, and AprilTags for ground truth.
- Offboard: an HP Z2 workstation (RTX 4000 Ada) for Isaac Sim and model fitting.

---

## Abstract
Robots that take commands like "go to #7" act on object memories that silently expire: objects get moved between mapping and command. Persistence models estimate how likely a remembered object is still in place, and systems revisit stale regions. Three things are missing:
- **the command-time decision** for a specific remembered instance: fly straight there, verify first from a cheap viewpoint, or search;
- a way to **recognise that instance among look-alikes** once found;
- evidence about **what a real detector can actually see** when the decision is made.

We contribute:
1. A **cost-aware trust / verify / search policy**. Its verify step picks the viewpoint, and the altitude where that helps, to maximise the probability of seeing the object per unit of time and energy.
2. **Map-belief re-identification:** the drone's belief about where each instance probably is now decides between look-alikes that appearance cannot separate.
3. A **six-week tracked-ground-truth benchmark** that tests persistence priors for command-time use.
4. A **calibrated perception chain** from simulator to Isaac Sim renders to a real open-vocabulary detector (YOLO-World + DINOv2). It is fitted per object size and viewing angle, and released with the code.

## 1. Origin
This work began with an audit of **ImpedanceGPT** (IROS 2025, arXiv 2503.02723). Seven hostile reviews and a simulation study later (§12), four lessons hold:
1. Claim only what prior work does not already hold.
2. Measure success against **physical ground truth**.
3. Pre-register outcomes that are still publishable if a hypothesis fails.
4. **Test every simulator assumption against renders and a real detector before believing it.** This lesson came out of this project: two of our own simulated claims shrank once tested (§6).

## 2. Related work and gap
| Work | What it does | What remains for us |
|---|---|---|
| Persistence filter (ICRA'16); TPM (ICRA'17); Bore et al. (T-RO'19); FreMEn | Learned survival / periodic presence; search for moved objects | Instance commands; action choice; look-alikes |
| **PreSIST** (2607.04057) | VLM → per-object survival prior fused with a persistence filter | **Baseline prior** inside our policy |
| Perpetua (IROS'25); PredictiveGraphs; FlowMaps; Patel & Chernova (CoRL'22) | Multimodal persistence; where moved objects go | Priors for the search phase |
| **Bogenberger et al.** (RA-L'26) | Per-instance stationarity, active revisits, LLM object-goal navigation | **Closest system.** No specific-ID commands, no trust/verify/search choice, ground robot |
| LT-Mem; VLMM; Memory-for-Attention | Which objects to re-observe under a budget (simulation) | Physical action cost; real perception |
| Quadrotor InstanceImageNav (2606.29917) | Drone flies to a specific instance | Stale memory; look-alikes |
| KnowNo (CoRL'23); ESC / L3MVN | Conformal ask-for-help; LLM search priors | Baselines |

**Gap (one sentence):** no work studies the **command-time choice between trusting, verifying and searching** for a **specific remembered instance**, together with **telling that instance apart from look-alikes** using the same belief, **under the error profile of a real detector**.

## 3. Research questions
| | Question | Primary output |
|---|---|---|
| **RQ1 Action** | Does a cost-aware trust / verify / search policy beat trust-then-search and verify-always? When does altitude help? | Success on true current ground truth; time and energy to success; per-class altitude effect |
| **RQ2 Identity** | How much does the map belief, used as a spatial prior, add to instance-level success when appearance cannot separate look-alikes? | Instance-exact vs intent-equivalent success, with vs without the spatial prior |
| **RQ3 Priors** | How well do persistence priors predict whether a specific object is still within 1 m of its mapped pose at command time? | Time-dependent Brier score; how prior errors change policy success |
| RQ4 Grounding (secondary) | Can language references be resolved jointly with persistence, with calibrated clarification? | Accuracy, risk–coverage, clarification rate |

**Headline (to confirm in flight):**
> "With realistic perception, appearance alone gets the right instance ~70 % of the time; adding the map belief raises this by ~20 points. Choosing trust / verify / search per command saves ~9 % of time (~11 % when the object moved). Flying higher pays off for small objects on desks, and not for objects on the floor."

## 4. System
1. **Mapping (ZED + Orin).**
   - Positional tracking and spatial mapping build the metric map.
   - An open-vocabulary detector proposes objects, and depth lifts them to 3D.
   - Each object gets an ID, class, appearance embedding, pose and `last_seen`.
2. **Persistence belief.** P(still within 1 m | class, Δt, time of day), from a hierarchical Weibull on activity hours with a return probability for periodic objects. Alternative priors are plugged in for comparison.
3. **Where it went.** A displacement distribution learned from E1, for the search phase.
4. **Policy.** Minimum expected cost (time + λ·energy) over the first viewpoint, then belief-greedy search (probability of seeing the object per unit cost):
   - **trust:** fly to the mapped pose;
   - **verify:** a viewpoint, and altitude, that sees the mapped pose cheaply;
   - **search:** the best viewpoint under the full belief.

   Detection probabilities come from the fitted real-detector model (§6.3), so the policy flies higher only where higher actually helps.
5. **Perception as proposals, identity by re-identification** (evidence in §6.3–6.4):
   - the detector gives *proposals*, because its class labels are unreliable;
   - metric size from depth rejects most scale confusions;
   - appearance (DINOv2) narrows a proposal to the right class;
   - the map belief decides between look-alikes.
6. **Safety floor.** An always-on depth CBF keeps clearance from obstacles and people.

## 5. Experiments
### E1: six-week persistence benchmark (RQ3)
- **Scope:** ≥ 60 tagged objects, ≥ 10 classes, ≥ 3 connected spaces, ≥ 6 weeks.
- **Ground truth:** AprilTags read by fixed cameras; tags masked in drone frames (with a masking ablation); ≥ 100 manual audits.
- **Analysis:**
  - Event = displacement > 1 m, or absence.
  - Priors compared: persistence filter, PreSIST-style, constant, human guesses, a local LLM.
  - Scoring: time-dependent Brier score with an object-cluster bootstrap.

### E2: command trials (RQ1, RQ2)
- **Main table:** replay of every arm on E1 logs, using measured flight times. The replay is validated against real flights.
- **Real flights:** ≥ 100 paired commands for trust / verify-always / ours, at 1 h and 1 day, across ≥ 3 spaces. Targets sit at several heights (floor, desk, shelf), and look-alike sets are included on purpose (identical chairs, mugs).

| Arm | Description |
|---|---|
| A1 | Trust-then-search |
| A2 | Verify-always |
| A3 | Persistence-filter prior + our policy |
| A4 | Commonsense / LLM prior + our policy |
| A5 | **Ours** |
| A5-low | Ours at 0.4 m only (ground-robot equivalent) |
| A5-nospatial | Ours without the spatial prior in re-identification |
| A6 | Oracle (true current pose) |

**Primary tests (pre-registered):**
1. A5 vs the better of A1 / A2 on time to success.
2. A5 vs A5-nospatial on instance-exact success.

Paired tests, with Holm correction for everything else.

### E3: perception calibration (supports RQ1, RQ2)
- Re-run the Isaac Sim chain (renders → YOLO-World → DINOv2 → fitted detection model) on a model of **our own lab**.
- Check the fitted model against ≥ 200 real drone frames with tag ground truth.
- Report the detection curve per size group and viewing angle, label confusions, and how much the size check corrects.

### E4: grounding under stale memory (RQ4, secondary)
Conformal clarification over IDs with persistence as input. Commands are written by ≥ 3 non-authors; about 50 are tested in flight. Kept small, since it is not the main claim.

## 6. Preliminary evidence (simulation and Isaac Sim, done)
All code and reports are in `sim/` and `isaac/`. The evidence has three layers:
- **Simulator:** a 6-week office world (64 objects, 11 classes) with Weibull activity-driven moves; a drone with real speeds and energy; 20 layouts × 12 seeds.
- **Isaac Sim:** the same scenes, 360 frames per layout × 3 layouts, with plain shapes and with realistic library models.
- **Real perception:** YOLO-World (28 prompts) and DINOv2 on those renders.

### 6.1 Policy (RQ1), holds at every level of realism
| Perception assumed | Ours vs trust: time | … moved objects | Ours vs verify-always | Ours − trust: success |
|---|---|---|---|---|
| Simple visibility model | −5 to −7 % | −17 to −20 % | −1.4 to −2.3 % | ≈ 0 |
| Detection calibrated on renders | −11 to −13 % | −15 to −19 % | −2 to −4 % | ≈ 0 |
| Fitted real detector | −7.6 to −9.5 % | −8 to −11 % | −2.4 to −4.1 % | −0.4 to −0.8 pts |
| Fitted real detector + repeating misses | [final run] | [final run] | [final run] | [final run] |

### 6.2 Identity (RQ2): our strongest result
- **Appearance is weak between look-alikes.** DINOv2 on Isaac Sim renders, 3 layouts:
  - right class first ~70 %;
  - AUC 0.88–0.89 against other classes;
  - **AUC 0.51–0.61 against same-class objects**, i.e. barely separable.

  Painting objects in flat colours did not help (0.59), because it removes the textures detectors rely on.
- **Simulator at render-measured appearance** (every class as hard to separate as the renders show):

  | | Map belief as spatial prior | Ours (instance success) | Trust |
  |---|---|---|---|
  | Only chairs alike | +6 to +7 points | — | — |
  | Every class alike (render-measured) | **+17 to +21 points** | 0.73–0.75 | 0.68–0.72 |

  Intent-level success stays at 100 %, and every remaining failure is a look-alike swap.

### 6.3 What a real detector sees
YOLO-World on realistic renders, 3 layouts, operating at ≤ 0.5 background false positives per frame:

| Finding | Number |
|---|---|
| Pixels needed for a 50 % chance of a correctly labelled detection | 270–340 px (the simulator had assumed 225) |
| Objects boxed under any label, when clearly visible | ≥ 87 % for most classes |
| Labels are unreliable | trash can → "cup" 63 %, stool → "chair" 77 %, monitor → "laptop" 35 % |
| Size from depth fixes confusions | picks the true class in 73 % of them (stool / chair 86 %, bin / cup 100 %) |
| Fitted model (size group × pixels × viewing angle) | beats a single curve on every held-out layout (log loss 0.61 → 0.56); looking steeply down hurts tall and medium objects |

### 6.4 Altitude (RQ1): narrowed from "altitude helps" to "altitude helps for some classes"
- **v7 claimed:** a drone's vantage point helps.
- **The simple visibility model said no:** at most 1.7 % time saved, so the pre-registered altitude rule failed.
- **Visibility calibrated on renders said yes,** but almost all of the gain came from laptops and mugs on desks.
- **The real detector, by object** (correctly labelled detection rate at 0.4 / 1.0 / 1.8 m):

  | Object | Open | Booth | Cubicle |
  |---|---|---|---|
  | Mug | 3 / 12 / 14 % | 2 / 6 / 6 % | 1 / 8 / 9 % |
  | Closed laptop | ~0 at every height | ~0 | ~0 |
  | Floor objects | slightly *worse* from 1.8 m | | |

- **Simulator with the fitted detector:** mugs are found ~40 % faster with altitude, while floor objects take 10–15 % longer. An apparent 50 % saving from laptops turned out to be an artifact: the simulator gave every repeated look from the same spot a fresh chance. With realistic repeating misses: **[final run]**.

**Revised claim:** choose altitude per class. The policy flies higher for small objects on raised surfaces, and stays low for floor objects.

## 7. Phase 2: moving targets (future work)
"Follow #12": ZED object tracking, Kalman prediction, CBF-bounded intercept, and re-identification after occlusion. Not in the first paper.

## 8. Contributions
1. **A cost-aware trust / verify / search policy** for instance commands on stale memory, with class-conditional vantage selection.
2. **Map-belief re-identification:** measured to matter most when appearance cannot separate look-alikes, which realistic perception shows is the common case.
3. **A six-week tracked-ground-truth benchmark** of persistence priors for command-time use.
4. **A calibrated perception chain** (simulator → Isaac Sim → real detector) with a fitted, viewing-angle-aware detection model, released as a reusable test bed.

## 9. Plan (remaining ~8 months)
| Month | Work |
|---|---|
| done | Simulator (Stages 1–2); Isaac Sim visibility + real-detector calibration (Stage 3a–3b); replay video |
| 0–1 | Ethics application; tag objects; external cameras; **model our lab in sim + Isaac Sim**; start E1 logging |
| 1–3 | On-drone mapping, IDs, perception chain (YOLO-World + DINOv2 + size check) on Orin; ROS 2 |
| 3–4 | E1 analysis; policy on real logs; E3 on real frames |
| 4–6 | Real flights (PX4; Isaac Sim + Pegasus for rehearsal) |
| 7–8 | Analysis, writing, dataset and code release |

## 10. Go/no-go (end of month 1)
| Check | Pass |
|---|---|
| Objects move | ≥ 15 events projected per class for ≥ 5 classes |
| Mapping + IDs | ≥ 80 % of tagged objects get a correct, stable ID |
| Real-frame detection | Fitted model within ±10 points of real detection rate per size group |
| Ethics | Approval filed |

## 11. Risks
| Risk | Mitigation |
|---|---|
| Policy gain is small (~9 %) | Lead with identity (RQ2, +17–21 pts); policy as the second contribution |
| Detector misses whole classes (closed laptops) | Report per class; try another detector (YOLOE / Grounding DINO) |
| Renders ≠ real images | E3 checks the fitted model on real drone frames before the flights |
| Few relocation events | More objects, a busier space, hierarchical pooling |
| Ethics delay | File in month 0; people-free camera views first |

## 12. Audit trail
| Version | Idea | Outcome |
|---|---|---|
| v1–v5 | VLM decision reuse / scheduling / staleness | Prior art or invalid evaluation |
| v6 | Object memory + navigation | False "first" claims; no reason for a drone |
| v7 | Aerial verify on stale object memory | Altitude as the headline claim |
| **v8 (this)** | **Trust / verify / search + map-belief identity, calibrated on real perception** | The altitude headline did not survive the real detector: kept per class. Identity became the strongest result. |

## 13. References (verify all; several from search snippets)
ImpedanceGPT 2503.02723 · PreSIST 2607.04057 · Persistence filter (Rosen et al., ICRA 2016) · TPM (Toris & Chernova, ICRA 2017) · Bore et al. T-RO 2019 · FreMEn · Perpetua 2507.18808 · PredictiveGraphs 2605.00121 · FlowMaps 2606.20209 · Patel & Chernova CoRL 2022 · Bogenberger et al. RA-L 2026 · LT-Mem 2608.19059 · VLMM 2607.16173 · Memory for Attention 2607.23797 · Quadrotor InstanceImageNav 2606.29917 · KnowNo CoRL 2023 · ESC 2301.13166 · YOLO-World (CVPR 2024) · DINOv2 (2023) · Isaac Sim 6.0
