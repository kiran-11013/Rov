# Hostile audit of proposal v3 (selective reuse)

**Verdict:** reject as written. P(accept) ≈ IROS 10 %, RA-L 5 %, CoRL 2 %.
**Reviewer summary line:** "An event-triggered VLM invocation scheme already published by 2026 VLA work and AESOP. The 'certificate' bounds a harm the authors define, which the always-on CBF already prevents and a distance-threshold trigger removes by construction."

## Kill shots
| # | Issue | Evidence | Fix |
|---|---|---|---|
| K1 | Novelty claim ("no work asks how long a decision remains valid") is false | AESOP (RSS 2024); React When You Need To (2609.22587); EAMP/PC-SET (2606.25629); Hi Robot, Helix, Fast-in-Slow, Plan Along the Way | Delete "first" claims; only the risk-controlled trigger threshold remains new |
| K2 | "Harmful staleness" is circular: it is defined from tracker state, so a distance trigger gets zero harm. The CBF floor makes it phantom. | v2 §3 CBF floor; harm = aggressive params while a human is near | Define harm by something geometry cannot see, or measure it physically (proxemic intrusion vs published comfort distance with mocap) |
| K3 | Certificate is empty at the planned n | 0 harms: UCB = 1 − 0.05^(1/n) → 1.98 % at n = 150, 7.2 % at n = 40; α = 1 % needs 299 episodes. Feedback loop breaks LTT in flight; handheld → flight shift. | ≥ 300 replayable episodes; report S1/S3 UCBs; call it an offline-replay guarantee, flight is an empirical check |

## Major flaws
1. A heuristic trigger probably catches ≥ 80–90 % of decision changes (geometric/track events) — estimate, not measured.
2. The killer experiment is rigged by the CBF. The advantage over pipelined always-call is ≤ ~0.5–1 call period.
3. Compute accounting: Jev every frame (70–500 ms + RTT) is in the same latency class as a 2–3B VLM on Orin.
4. "Unseen category" is unfalsifiable: the open-vocab detector only emits prompted strings; condition on correct detection.
5. Trained-router baseline set up to lose (one-hot). Use: embedding + LR, AESOP-style kNN, local Qwen 0.5–3B yes/no logprob.
6. ≡ is ill-defined for continuous outputs; measure the VLM self-flip noise floor.
7. "Certified scheduler" is a scalar threshold sweep.
8. ImpedanceGPT reintroduced as a host after two audits called it a strawman — drop it.
9. Make-or-break data too small (≈ 7 frames/episode); need ≥ 1,000 frames.

## What survives
1. Fixed-sequence LTT on a trigger under temporal correlation, with replay vs flight validation.
2. Empirical decomposition of why VLM decisions change (geometric / semantic / self-inconsistency).
3. Pre-registered negative-result path.
4. Cost-weighted harm, once it is non-circular.

## Reframings
| | Paper | Jev role | Risk |
|---|---|---|---|
| A | "Why do VLM decisions in robot controllers go stale?" — measurement + trigger Pareto study | One of four triggers | Medium |
| B | Two-stage trigger: heuristic for geometry, Jev only for the semantic-only residual | Main method | High |
| C | Workshop: hosted System-One model in a drone loop — latency, calibration drift, failure modes | Whole subject | Low |

**Reviewer's bottom line:** the Jev angle alone cannot carry a top-venue paper.

## Next-week experiment (no Jev, no flight)
30 handheld ZED episodes → host VLM on every frame offline → equivalence classes + harm labels → VLM self-flip rate on 200 static frames → sweep the heuristic trigger S3.
**Pass:** ≥ 20 % of harmful changes missed by S3 (at ≥ 95 % geometric-event recall), and self-flip rate < ⅓ of the true change rate.
**Fail:** pivot to A with S3 as the method, or to C.

Closest prior work: AESOP https://www.roboticsproceedings.org/rss20/p114.pdf · React When You Need To https://arxiv.org/abs/2609.22587 · EAMP https://arxiv.org/abs/2606.25629 · Hi Robot https://arxiv.org/abs/2502.19417 · Helix https://www.figure.ai/news/helix · Fast-in-Slow https://arxiv.org/abs/2506.01953 · KnowNo https://robot-help.github.io/ · Plan Along the Way https://arxiv.org/abs/2608.28075 · VLA-Cache https://arxiv.org/abs/2502.02175 · AlphaAdj https://arxiv.org/abs/2603.21142 · ASMA https://arxiv.org/abs/2409.10283
(Reviewer worked from search snippets; verify every citation before use.)
