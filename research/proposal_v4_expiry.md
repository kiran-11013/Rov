# When Do VLM Decisions Expire? Measuring and Predicting Decision Staleness in VLM-Driven Drone Control

**Status:** v4. Redesigned from scratch as a measurement + benchmark paper (Option A of `audit_v3.md`). Every flaw from the three audits is addressed in §4.
**Target:** IROS / RA-L. **Platform:** 1–2 kg PX4 quad, ZED stereo, Jetson Orin, mocap/AprilTags.

---

## 1. Motivation
VLMs now sit inside robot control loops and set semantic decisions: safety margins (AlphaAdj, ASMA), planner or impedance parameters (EAMP, ImpedanceGPT), waypoints (See-Point-Fly). Every such system has to decide **when to re-query the VLM**. Fixed rates, event triggers (React When You Need To), embedding-based gates (AESOP) and dual-system architectures (Hi Robot, Helix, Fast-in-Slow) all embed an *assumption* about how fast and why VLM decisions go stale. **Nobody has measured it.**

## 2. Research questions
| | Question |
|---|---|
| **RQ1 Rate and cause** | How often do VLM-controller decisions change in real scenes, and what share is due to geometric change, semantic change, or VLM self-inconsistency? |
| **RQ2 Predictability** | Which cheap signals catch the harmful changes, and at what compute? Compare trigger families at matched call rate. |
| **RQ3 Consequence** | Does acting on a stale decision measurably reduce human clearance or comfort in flight, and what does the best trigger cost in energy to prevent it? |

## 3. Method
1. **Data.**
   - ≈ 300 handheld ZED episodes at drone height (2–3 indoor environments).
   - **Counterfactual scene pairs:** geometry fixed with semantics changed (person ↔ mannequin, sign added, person picks up a ladder), and semantics fixed with geometry changed.
   - 40 flight episodes. Released as the **Expiry** dataset.
2. **Dense oracle replay.**
   - Run each host × VLM on **every frame** offline (RTX 4090) to get the decision at every timestep.
   - Any trigger policy can then be evaluated *exactly*, with no closed-loop feedback.
3. **Behavioural equivalence.** Decisions d, d′ are equivalent if the host controller's output under them differs by ≤ ε over horizon H.
4. **Change taxonomy.**
   - Each non-equivalent change is attributed as geometric, semantic or self-inconsistency.
   - Attribution uses the counterfactual pairs (causal) plus the re-run flip rate on static frames (noise floor).
5. **Harm, measured downstream.**
   - Replay the controller with the stale vs the fresh decision.
   - Harm = Δ minimum clearance to humans, and violation of a published human–drone comfort distance (≈ 1.2–1.6 m).
   - Not computable from tracker state alone, so not circular.
6. **Trigger benchmark** at matched call rates:

   | Family | Example |
   |---|---|
   | Fixed rate | every k frames |
   | Geometric heuristic | new/lost track, label change, Δdistance/Δvelocity > θ |
   | Image-feature change | vision-encoder feature distance (React-When-You-Need-To style) |
   | Text-embedding gate | MiniLM/e5 + LR; AESOP-style kNN |
   | Small local LLM | Qwen 0.5–3B yes/no with logprobs, temperature-scaled |
   | Typed calibrated model | Jev (hosted); results conditioned on correct detection |

   Metrics: harmful-change recall, detection delay, false-trigger rate, latency and **joules per frame including the always-on perception**; cloud cost reported separately.
7. **Flight validation.**
   - Mannequin on a cart under mocap; scripted intrusions at varied distances and speeds.
   - Arms: best trigger, geometric heuristic, pipelined always-call.
   - Report clearance **distributions** and comfort-zone violations; check that offline rankings transfer.

**Hosts.**
- (1) VLM-tuned CBF safety margin (AlphaAdj/ASMA style).
- (2) VLM planner-parameter retuning (EAMP style).
- ImpedanceGPT dropped.

**VLMs.** Two families (e.g. Molmo-7B, Qwen-VL-3B).

## 4. Audit flaws → design response
| Flaw | Response |
|---|---|
| False novelty (K1) | No triggering-novelty claim; prior triggers become benchmark arms |
| Circular harm (K2) | Downstream behavioural harm vs published comfort distance |
| Empty certificate (K3) | Dropped as headline; episode-level bootstrap CIs; bound only as secondary, replay-validated |
| Feedback loop in LTT | Dense oracle replay evaluates triggers exactly; flight = validation |
| Heuristic may win | It is a reported finding, not a threat |
| Geometry vs semantics unclear | Counterfactual scene pairs give causal attribution |
| VLM self-inconsistency | Measured noise floor, subtracted |
| ≡ undefined for continuous outputs | Behavioural equivalence |
| Strawman host | Two in-flight hosts; ImpedanceGPT dropped |
| Single-model findings | Two VLM families |
| Rigged Jev comparison / unseen categories | Strong text baselines; detector recall reported; Jev conditioned on detection |
| Cloud compute | Joules/latency per frame incl. perception; cloud separate |
| Rigged flight test | Mannequin cart + mocap, clearance distributions, pipelined always-call arm |
| Small data | ≈ 300 episodes + 40 flights, released |

## 5. Contributions
1. The **first measurement** of decision staleness in VLM-driven robot controllers (rate, causes, physical consequence) across 2 hosts × 2 VLMs.
2. **Dense oracle replay with counterfactual scene pairs:** an exact, causal evaluation protocol for re-query triggers.
3. A **matched-call-rate ranking** of trigger families, including prior-work-style gates and Jev, on recall, latency and energy.
4. **Flight validation** and the open **Expiry** dataset and benchmark.

**By-product (workshop paper):** latency, timeout and calibration behaviour of a hosted System-One model (Jev) in a real drone loop.

## 6. Plan (6–9 months)
| Month | Work |
|---|---|
| 1 | Pilot (30 episodes) + host re-implementation → go/no-go |
| 2–3 | Full recording + counterfactual pairs + dense replay |
| 4 | Taxonomy, harm replay, trigger benchmark |
| 5–6 | Flights, analysis |
| 7–9 | Writing, dataset release, submission |

## 7. Go/no-go (pilot)
**Continue if both hold:**
- harmful decision changes occur often enough to matter (≥ a few per 10 min of footage);
- VLM static self-flip rate < ⅓ of the true change rate.

The geometric/semantic split is **reported, not gated**.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Changes are rare in natural footage | Counterfactual pairs and scripted intrusions guarantee coverage; natural-footage rate reported separately |
| Self-inconsistency dominates | That is itself a headline finding (VLM controllers are noisy), and motivates temporal smoothing |
| Host re-implementations differ from the originals | Release code; validate against published behaviour where possible |
| No Jev access | Jev is one arm; the paper stands without it |

## 9. Positioning (verify all citations)
AESOP (RSS 2024) · React When You Need To (2609.22587) · EAMP/PC-SET (2606.25629) · Hi Robot (2502.19417) · Helix · Fast-in-Slow (2506.01953) · VLA-Cache (2502.02175) · Reducing temporal redundancy for VLA inference (2607.12287) · KnowNo · AlphaAdj (2603.21142) · ASMA (2409.10283).
