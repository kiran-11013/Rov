# REFLEX — Risk-controlled Escalation From Lightweight EXperts
### Compute-efficient VLM-driven impedance control with calibrated System-One decisions

**Status:** v1, pre-audit · **Target:** IROS 2027 / RA-L · **Platform:** 1–2 kg PX4 quad, ZED stereo, Jetson Orin

---

## 1. Problem

VLM-in-the-loop robot controllers (ImpedanceGPT [IROS'25], SwarmVLM, ImpedanceDiffusion, VLM-tuned CBFs such as AlphaAdj/ASMA) use a large generative VLM to turn an image into a **control decision** (a scenario label, an impedance parameter set, a safety margin). They pay generative cost for a discriminative job:

- ImpedanceGPT runs Molmo-7B (4-bit), lets it generate a long free-text scene description (`max_new_tokens=2000`), regex-parses it, embeds it with MiniLM, and retrieves the nearest of 20 scenarios with FAISS. Reported latency **5–8 s per decision**, 80 % accuracy, on an RTX 4090.
- Most of that time is autoregressive decoding of text that is immediately thrown away after embedding.
- The output carries **no usable confidence**, so the system cannot tell when it is wrong.

This makes VLM-driven controllers too slow for re-planning during flight and too expensive to run onboard.

## 2. Gap

1. Nobody separates **perception** (what is where, which the VLM is needed for) from **decision** (which parameter set to use, a typed classification) in VLM-driven controllers.
2. Nobody gives a **guarantee** that a cheaper pipeline preserves the decision accuracy of the expensive one. Speed-ups are reported as averages, without a bound on the accuracy lost.
3. Calibrated decision models (TypeSafe Jev, released Sept 2026: text/JSON in, typed Choice/Score/Boolean out with calibrated probabilities, one parallel pass, 70–500 ms) have never been evaluated as the decision layer of a robot controller.

## 3. Research question

> **Can a VLM-driven impedance controller hand its decision step to a calibrated System-One model, escalating to the full VLM only when that model is unsure, and match the original decision accuracy, with a finite-sample guarantee, at a fraction of the compute?**

| ID | Sub-question | Hypothesis |
|---|---|---|
| RQ1 | Accuracy | REFLEX at its chosen threshold is **non-inferior** to ImpedanceGPT (margin ε = 3 points), paired test on ≥ 400 labelled scenes. |
| RQ2 | Compute | ≥ 5× lower median end-to-end decision latency and ≥ 5× fewer on-device GPU-seconds per decision. |
| RQ3 | Calibration | Jev's probabilities are calibrated enough on this domain (ECE ≤ 0.05) for a confidence gate to beat a random gate at every coverage level. |
| RQ4 | Closed loop | In flight, REFLEX keeps clearance and success rates equal to ImpedanceGPT while re-deciding ≥ 5× more often. |

## 4. Method

```
ZED RGB-D frame
  │
  ├─(1) PERCEIVE-BRIEFLY  Molmo-7B in pointing mode → obstacle points + type tokens (≈20–50 output tokens, not ~500)
  │
  ├─(2) COMPUTE-IN-CODE   counts, pairwise spacing (metric, from ZED depth), before/after-gate, clearance → JSON scene state
  │                       (Jev is weak at counting/arithmetic; all arithmetic is deterministic code)
  │
  ├─(3) DECIDE-TYPED      Jev: Choice over K scenario/parameter classes (+ Boolean "scene is in-distribution") → p(·)
  │
  ├─(4) GATE              if max p ≥ τ*: act on the argmax parameters
  │                       else: ESCALATE → full ImpedanceGPT pipeline (verbose VLM + RAG)
  │
  └─(5) EVENT-TRIGGER     re-query only when the JSON state changes (new/removed obstacle, spacing class flips)
```

**Risk-controlled threshold (the guarantee).** τ\* is chosen on a held-out calibration set with Learn-then-Test / conformal risk control, so that

  P( accuracy(REFLEX_τ\*) ≥ accuracy(ImpedanceGPT) − ε ) ≥ 1 − δ

under exchangeability of calibration and test scenes. This turns "about the same accuracy" into a certified statement, and the τ sweep gives the full **cost–accuracy Pareto curve**.

**Fully local variant.** Same pipeline with `open-jev` (open reproduction, on-device via Transformers.js/WebGPU) or a distilled small classifier in place of the hosted Jev. This answers "you just moved compute to the cloud".

## 5. Experimental design

**Datasets**
- **D-Top:** top-down scenes reproducing ImpedanceGPT's 10 hard + 10 soft layouts, expanded to ≥ 200 images with varied lighting, counts and spacing.
- **D-Ego:** ≥ 300 egocentric ZED RGB-D frames from the drone. Humans, stands, gates, mixed scenes, dim light.
- Labels: ground-truth scenario class, plus the optimal parameter set (from the simulation tuning procedure).
- Splits: train-free (no fine-tuning) · calibration 30 % · test 70 %. Scenes are split by **episode**, not by frame.

**Arms**

| Arm | Pipeline | Purpose |
|---|---|---|
| A0 | ImpedanceGPT as published (verbose Molmo → MiniLM → FAISS) | Baseline to match |
| A1 | Pointing Molmo + code features + FAISS (no Jev) | **Separates the saving from shorter prompts from the saving from Jev** |
| A2 | Pointing Molmo + code features + Jev (no gate) | Core decision layer |
| A3 | A2 + risk-controlled gate → A0 | **Full REFLEX** |
| A4 | A3 with open-jev / distilled classifier, fully on-device | Fully local, reproducible |
| A5 | YOLO + depth + Jev (no VLM) | Cheapest lower bound |
| A6 | A0 with Molmo prompted to output JSON directly (short) | Strong "just prompt better" baseline |

**Metrics**
- Decision accuracy (scenario class), parameter error (L1 to optimal set).
- Latency: median and p95, split into on-device and network parts.
- On-device GPU-seconds and Jetson energy (from the INA power rails); GPU memory.
- $ per decision.
- Escalation rate; ECE; risk–coverage curve.
- Closed loop: success rate, minimum clearance, decisions per second, behaviour on Jev timeout.

**Statistics**
- Paired non-inferiority test (McNemar-based, ε = 3 pts, one-sided α = 0.05), plus bootstrap CIs on all latency and energy numbers.
- Closed loop: ≥ 20 flights per arm (A0, A3, A4) over 4 scenario families.

**Platform**
1–2 kg PX4 quad, ZED 2i, Jetson Orin NX/AGX. Mocap or AprilTags for ground truth. A0 runs offboard on an RTX 4090 (as in the original paper) and also on the Orin, where it is feasible.

## 6. Expected contributions

1. **A decomposition principle** for VLM-driven controllers: *perceive briefly → compute in code → decide typed → escalate when unsure.*
2. **A finite-sample guarantee** that the accelerated pipeline does not lose more than ε accuracy against the full VLM pipeline, via risk-controlled gating.
3. **The first evaluation** of a calibrated System-One decision model as a robot-control decision layer, including whether its calibration holds out of domain.
4. **Closed-loop flight evidence** that faster decisions allow re-deciding during flight, demonstrated on two VLM-driven pipelines (ImpedanceGPT plus one VLM-tuned safety-margin pipeline).
5. An open benchmark: D-Top / D-Ego with labels, code, and all arms.

## 7. Known risks (pre-declared)

| Risk | Mitigation |
|---|---|
| Jev is waitlist-only; no weights | Build A4 first; apply for access now; the method is model-agnostic |
| Network latency / dropouts on a drone | Timeout → escalate or hold the last safe parameters; report the timeout rate |
| Most of the speed-up may come from shorter prompts, not Jev | A1 and A6 isolate it; report it honestly even if Jev's share is small |
| ImpedanceGPT's 20-scenario task is too easy, so every arm saturates | Expand to harder, mixed and egocentric scenes (D-Ego) and a second pipeline |
| Calibration claims are vendor-reported | RQ3 measures them; the gate falls back to empirical recalibration (temperature / isotonic) |

## 8. Open questions for the reviewer

1. Is the novelty sufficient given LLM cascades/routing (FrugalGPT, RouteLLM, learning-to-defer) and VLM token-pruning work?
2. Is the risk-control guarantee meaningful here, or decorative?
3. Is ImpedanceGPT a strong enough host task, or a strawman baseline?
4. What is the single experiment that would make or break this paper?
