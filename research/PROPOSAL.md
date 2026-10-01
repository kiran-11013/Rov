# Is #7 Still There? Acting on Expiring Object Memory for Drone Navigation to Remembered Objects

**One line:** A drone maps a space and gives every object an ID. Later you say *"go to #7"* or *"go to the chair by the door"*. By then #7 may have moved. We measure how fast object memories expire and build a drone that decides whether to trust, verify or search before it flies.

**Status:** v6 draft (pre-audit). Replaces v5 (`proposal_v5_halflife.md`); v5's "when to re-ask the VLM" becomes the *verify* action here.
**Author:** M.Tech, IIT Hyderabad · **Target:** IROS 2027 (primary), RA-L (alternative)
**Platform:** 1–2 kg PX4 quadrotor, ZED stereo (positional tracking, spatial mapping, object and body tracking), mono cameras, LiDAR, IMU, AprilTags / mocap for ground truth, RTX 4090, Jetson Orin

---

## Abstract
Language-commanded robots increasingly navigate to objects stored in a semantic map. Between mapping and command, objects get moved, and the robot's memory silently expires. Existing dynamic scene graphs update an object only when the robot sees it change, and aerial object-goal navigation searches for objects never seen before. We study the gap between the two: **navigating to a specific remembered object whose current location is uncertain**.
- We measure per-class **object-memory half-lives** from weeks of ground-truth logs in a real lab, and test whether language models can predict them for unseen classes.
- We build a drone that, given a command by ID or by language, estimates the probability that its memory is still valid. It then picks the action with the lowest expected time: **trust** (fly to the last-known position), **verify** (re-observe from a vantage point first) or **search** (belief-guided). It re-identifies the object to keep its ID stable.
- Language commands are grounded to IDs with calibrated confidence; the drone **asks for clarification** when unsure.

Phase 2 extends the system to moving targets (track and intercept).

## 1. Origin
This line of work began with an audit of **ImpedanceGPT** (IROS 2025, arXiv 2503.02723):
- The VLM ran once offline and picked 1 of 20 near-duplicate scenarios.
- The obstacle type was passed via the filename.
- There were no baselines.

Four hostile reviews of follow-up ideas (audit trail in §11) taught three lessons:
1. Do not claim novelty that prior work already holds.
2. Define harm and success by **physical ground truth**, not by quantities the system itself computes.
3. Design so that **any outcome is publishable**.

## 2. Problem and gap
| Area | Examples | Covers | Missing |
|---|---|---|---|
| Dynamic open-vocabulary scene graphs | DovSG (RA-L 2025), DynaMem, LOST-3DSG | Update the map **when change is observed** | What to do about objects **not seen recently** |
| Aerial object-goal navigation | UAV-ON, AirHunt, spatial belief fields (2026) | Finding objects **never seen** | Going to a **specific remembered instance** |
| Lifelong / probabilistic object maps | Stationarity scores, decaying object maps | Confidence decay over time | Decay rates **hand-set**; ground robots; no decision policy for a flying robot; no language grounding |
| Motion-aware semantic maps | Vision–Language–Motion Maps (2607.16173) | A queryable motion attribute in the map | **Closest work — must be read in full and positioned against** |

**Gap:** no work measures how fast object memories expire per class in real spaces, and then uses those rates to decide, at command time, whether a drone should trust, verify or search for a specific remembered object, with commands given by ID or by language.

## 3. Research questions
| | Question | Primary output |
|---|---|---|
| **RQ1 Memory half-life** | How long does each object class stay where it was last seen in a real, in-use indoor space? Can language models predict this for unseen classes? | Kaplan–Meier survival curves per class; zero-shot half-life prediction error |
| **RQ2 Acting on expiring memory** | Does a half-life-aware trust / verify / search policy beat always-trust and always-search on success and time? | Success within 1 m of the **true current** position; time-to-success; wasted trips |
| **RQ3 Grounding commands** | How accurately can language commands be resolved to map IDs with calibrated confidence, and how often must the drone ask? | Accuracy, calibration (ECE), risk–coverage, clarification rate |

**Headline we aim for:**
> "Chairs keep their place for about X minutes and people for Y seconds. Using these half-lives, the drone reaches the right object Z % more often, W % faster, than trusting its map."

## 4. System
1. **Mapping.**
   - ZED positional tracking + spatial mapping (LiDAR-aided) build the metric map.
   - An open-vocabulary detector proposes objects; depth lifts them to 3D; instances are associated across views.
   - Each object gets an **ID**, class, VLM-written description, position, covariance and `last_seen`.
2. **Memory model.**
   - P(object still at last position | class, Δt) = S_c(Δt), the survival curve estimated in RQ1.
   - For unseen classes, the hazard is predicted zero-shot from the class name and description, then shrunk toward the nearest known class.
   - A displacement prior (empirical: same room, near similar furniture) says where a moved object probably went.
3. **Decision policy.** Pick the action with minimum expected time to success, given P:
   - **Trust:** fly to the last-known position; fall back to search if absent.
   - **Verify:** fly to a vantage point, re-observe, then commit. *(This is where v5's "when to re-ask the VLM" lives.)*
   - **Search:** belief-weighted coverage over likely locations.
4. **Re-identification.** Appearance embedding (e.g. DINOv2) + VLM description + spatial prior keep ID #7 = #7 after relocation. Identical-looking objects are a stress test.
5. **Command grounding.**
   - By ID: direct.
   - By language: a resolver returns a probability distribution over map IDs. If the top probability < τ or the margin is small, the drone asks a clarifying question ("#7 or #9?").
6. **Safety floor.** An always-on depth-based CBF keeps clearance from obstacles and people.

## 5. Experiments
### E1 Persistence study (RQ1)
- Fixed external cameras (not the drone) observe ≈ 30 tagged objects over **2–4 weeks** of normal lab use: chairs, carts, bags, boxes, bins, laptops, plus people.
- Kaplan–Meier survival per class, censored at the end of observation.
- Zero-shot prediction of class half-lives for **held-out classes**: VLM vs small local LLM vs Jev vs embedding regression.

### E2 Command trials (RQ2)
- ≈ 200 commands issued 1 min, 10 min, 1 h and 1 day after mapping.
- Ground truth: the object's **true current** position from external cameras / mocap. The drone never sees the tags (placed out of its view or on the underside).
- Moved objects arise both naturally (E1 logs) and from scripted relocations; the two are reported separately.

| Arm | Description |
|---|---|
| A1 Trust map | Always fly to the last-known position |
| A2 Always search | Ignore memory, search from scratch |
| A3 Uniform decay | Same half-life for every class |
| A4 Ours | Class-specific half-life + trust/verify/search + re-ID |
| A5 Oracle | Knows the true current position (upper bound) |

**Metrics:** success (within 1 m of the true current position, within a time budget), time-to-success, path length, wasted trips, ID errors, minimum clearance to people, battery used.

### E3 Language grounding (RQ3)
- ≈ 300 language commands over recorded maps, including deliberately ambiguous ones ("the chair" when there are three).
- Resolvers: Jev (typed Choice over IDs), small local LLM with log-probabilities, embedding similarity.
- Metrics: accuracy, ECE, risk–coverage, clarification rate vs accuracy trade-off.
- A subset is run in flight.

### Statistics
- **Unit:** the command, clustered by session; cluster-bootstrap CIs.
- **Primary test (pre-registered):** A4 vs the best of A1–A3 on success within the time budget.
- Holm correction for all secondary comparisons.

## 6. Phase 2: moving targets (extension, not in the first paper)
"Follow #12" / "go to #12" while #12 moves:
- Track with ZED object and body tracking + Kalman prediction.
- Intercept with a safety distance enforced by the CBF.
- Re-ID after occlusion.
- Metrics: time to intercept, tracking loss, minimum distance to people (proxemic sweep 0.6–2.0 m).

## 7. Role of Jev
Jev (TypeSafe AI, Sept 2026) is a hosted, text-only model returning typed answers with calibrated probabilities. Its two roles here are both natural fits:
1. **Grounding:** a typed Choice over map IDs (≤ 255 options) with calibrated confidence. This drives clarification.
2. **Zero-shot mobility:** a typed Score / Choice on how likely a class is to move within Δt.

Jev is always compared against a small local LLM and an embedding baseline. Version pinned, all I/O logged. If access never comes, both roles are filled by the local LLM and Jev is dropped without affecting the paper.

## 8. Contributions
1. **Object-memory half-lives:** the first per-class measurement of how long remembered object positions stay valid in a real indoor space, plus zero-shot prediction for unseen classes.
2. **A half-life-aware trust / verify / search policy** for drones navigating to remembered objects, with re-identification.
3. **Calibrated language-to-ID grounding** with clarification, comparing typed decision models, local LLMs and embeddings.
4. **Real-flight evaluation** and an open dataset (persistence logs, maps, commands, ground truth).

## 9. Plan (9 months)
| Month | Work | Gate |
|---|---|---|
| 1 | Install external cameras + tags, start E1 logging; mapping pipeline on the drone | Go/no-go (§10) |
| 2–3 | Object IDs, descriptions, re-ID; memory model; policy in simulation | |
| 3–4 | E1 analysis; grounding resolvers; E3 offline | |
| 5–6 | E2 flight trials | |
| 7 | Analysis; Phase 2 prototype if time allows | |
| 8–9 | Writing, dataset release, submission | |

## 10. Go/no-go (end of month 1)
| Check | Pass |
|---|---|
| Objects actually move in the chosen space | ≥ 20 % of tracked objects relocate at least once per week |
| Mapping + ID assignment works | ≥ 80 % of tagged objects get a correct, stable ID in one mapping pass |
| Re-ID of moved objects | ≥ 70 % correct on a small scripted test |

**If objects rarely move:** move logging to a busier space (common room, workshop), or rely on scripted relocations and report the natural rate honestly.

## 11. Risks
| Risk | Mitigation |
|---|---|
| Lab objects rarely move | Busier space; scripted relocations; natural rate reported |
| Identical objects break re-ID | Stress test reported; spatial prior + description; ask the user when ambiguous |
| Overlap with VL Motion Maps / stationarity-score work | Read in full; position on measured half-lives + drone decision policy + grounding |
| Ground truth leaks into perception | Tags out of the drone's view; GT from external cameras/mocap only |
| No Jev access | Local LLM fills both roles |

## 12. Audit trail
| Version | Idea | Outcome |
|---|---|---|
| Two Witnesses | Depth + VLM cross-check | Wrong harm direction; prior art |
| v1 REFLEX | Jev replaces the VLM decision | Decision tree matched Jev |
| v2 Saccade | Jev schedules VLM calls | Certified cascades exist |
| v3 Selective reuse | Certified reuse of VLM decisions | False novelty; circular harm |
| v4 Expiry | Staleness measurement | Open-loop replay invalid; latency ignored |
| v5 Half-life | Decision half-life study | Est. IROS 45–50 % if executed; replaced by the author for a task-driven design |
| **v6 (this)** | **Expiring object memory + navigation to remembered objects** | Pending audit |

## 13. References (verify all)
ImpedanceGPT 2503.02723 · DovSG 2410.11989 · DynaMem · LOST-3DSG 2601.02905 · Vision–Language–Motion Maps 2607.16173 · UAV-ON 2508.00288 · AirHunt 2601.12742 · Spatial Belief Fields 2609.05841 · Dual-layer belief mapping 2609.08164 · Lifelong semantic maps 2010.08846 · Semantic Linking Maps 2006.10807 · TypeSafe Jev (Sept 2026)
