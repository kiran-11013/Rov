# Is It Still True? Certified Selective Reuse of Foundation-Model Decisions for Efficient Embodied Control

**Status:** v3 (research framing). Builds on `saccade_proposal_v2.md`; method details and arms are unchanged there.
**Target:** IROS / RA-L, with a CoRL-level stretch.

---

## 1. Problem
Robots increasingly run foundation models (VLMs, LLMs, VLAs) inside the control loop to make **semantic decisions**: how cautious to be, what an obstacle is, which controller parameters to use. On edge hardware these models cost seconds per call, but the world changes slowly, so most of the time the previous decision is still correct. Current systems either call the model continuously (late reactions, high energy) or once per scene (blind to change). **No work asks how long a foundation-model decision remains valid, or how a robot can know that cheaply and safely.**

## 2. Research question
> **Can an embodied agent predict, from cheap observations, whether its last foundation-model decision is still valid, with calibration good enough to certify a bound on harmful staleness, and how much foundation-model compute does this save?**

## 3. Concept: selective reuse
| Known | Ours |
|---|---|
| Selective prediction: abstain when unsure | **Selective reuse**: reuse the last expensive decision only when confident it still holds |
| Risk = wrong on this input | Risk = acting on a decision that has become stale **in a harmful way** |

**Formulation.** At step *t* the robot holds decision *d* = F(o_τ) from the expensive model F at time τ ≤ t, and a cheap state s_t (detector + depth + tracker JSON). A validity predictor g(s_t, s_τ) → p̂_t estimates P(F(o_t) ≡ d), with ≡ defined up to decision-equivalence classes. The robot reuses *d* iff p̂_t ≥ λ, else it calls F. Objective:

  min E[# calls to F]  s.t.  P(reuse ∧ stale ∧ harmful) ≤ α

λ is chosen by risk-controlled thresholding (Learn-then-Test, Clopper–Pearson) on **episode-level** calibration data, with cost-weighted harm (e.g. aggressive parameters kept while a human is near).

## 4. Sub-questions
| ID | Question | Evidence |
|---|---|---|
| RQ1 Predictability | Can validity be predicted from cheap perception? | Validity-predictor accuracy; selective-risk (AURC) curve |
| RQ2 Generalization | Do typed, calibrated decision models (Jev) predict validity better than trained classifiers, especially on **unseen object categories**? | Jev vs LR / tree / kNN on held-out categories, episode-bootstrap CIs |
| RQ3 Certified efficiency | How much F-compute is saved at a certified bound, and does it change behaviour in flight? | Call reduction at α ∈ {1, 2, 5}%; success vs hazard-onset distance in dynamic scenes |

## 5. Method (summary; details in v2)
1. **Always on:** open-vocab detector + ZED depth + tracker → JSON state with open-vocabulary object text and a delta since the last F call.
2. **Validity predictor:** typed Booleans ("still valid?", "anything unaccounted for?") with calibrated probabilities. Instantiated by Jev, an open local typed model, and trained-classifier baselines.
3. **Certified threshold** via risk control on episodes.
4. **Safety floor:** an always-on depth CBF, independent of the semantic layer.
5. **Hosts:** (a) a VLM-tuned safety-margin controller that runs in flight; (b) an ImpedanceGPT-style impedance-parameter selector.

## 6. Evaluation
- **Data:** ≥ 150 handheld ZED episodes at drone height (calibration) + ≥ 40 flight episodes (validation) + held-out object categories.
- **Baselines:** always-call, call-once, fixed-period, heuristic trigger, trained router, short-JSON-prompted VLM.
- **Metrics:** F-call rate; harmful-staleness rate and its UCB; decision accuracy on equivalence classes; latency median/p95 (on-device vs network); Jetson energy; $/decision; closed-loop success and minimum clearance.
- **Killer experiment:** a person steps into the path at varying distances and speeds. Always-call fails from latency, call-once fails from staleness, selective reuse succeeds at a fraction of the compute.

## 7. Contributions
1. **Selective reuse:** problem formulation, with harmful staleness as the risk.
2. A **certified scheduler** for foundation-model calls in embodied control.
3. The **first study** of calibrated typed decision models (Jev) versus trained classifiers for semantic validity under unseen object semantics.
4. **Real-flight evidence** on two VLM-driven controllers, plus an open dataset.

## 8. Why this is robust
- The question is model-agnostic. Jev is a tested hypothesis (RQ2), not a dependency. A negative RQ2 is still a reported finding.
- It generalises to any robot with a foundation model in the loop. The drone is the testbed.

## 9. Make-or-break (first 2 weeks)
1. Profile VLM prefill/decode on 4090 and Orin.
2. Collect ~300 labelled frames from ≥ 40 handheld episodes, including held-out categories.
3. Test RQ1 + RQ2:
   - Validity predictable at useful coverage? If not, rethink the cheap state.
   - Jev beats trained routers on unseen categories? If not, RQ2 becomes a negative finding and the paper stands on RQ1 + RQ3.
