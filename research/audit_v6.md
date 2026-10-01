# Hostile audit of proposal v6 ("Is #7 Still There?")

**Verdict as written:** IROS 15–20 %, RA-L 8–12 %, CoRL ≤ 3 % — a regression from v5.
**If all v7 changes are made:** IROS 40–45 %, RA-L 25–30 %, CoRL ≤ 8 %.
**Reviewer summary line:** "A drone re-run of temporal persistence modelling (Toris & Chernova 2017) and the persistence filter (Rosen et al. 2016), with zero-shot persistence priors already published by PreSIST (2026); the decision policy is a hand-built threshold rule, it is underpowered, and the drone adds risk without adding drone-specific science."

## Kill shots
| # | Issue | Fix |
|---|---|---|
| K1 | "First per-class measurement" + zero-shot half-life is false: TPM (ICRA'17), persistence filter (ICRA'16), Bore et al. (T-RO'19), FreMEn, PreSIST (2607.04057: VLM → Weibull survival prior + persistence filter) | RQ1 becomes a calibration benchmark of existing priors on tracked ground truth; PreSIST as a baseline |
| K2 | Gap table false ("hand-set decay", "update only when observed"): Bogenberger et al. RA-L'26, PredictiveGraphs | Gap = command-time trust/verify/search for a specific instance, with an aerial vantage, under language references that may be stale |
| K3 | In one room, trust ≈ verify (flying there re-observes the object) | ≥ 3 connected spaces with occluders; explicit verify cost; "verify always wins" pre-registered as publishable |

## Major flaws → fixes
1. RQ1 starved (≈ 4 of 6 classes < 10 events) → ≥ 60 objects, ≥ 10 classes, ≥ 6 weeks, ≥ 15 events per reported class, hierarchical Weibull, RMST at 1 h / 1 d.
2. "Half-life" wrong model (periodic, returning objects) → event = displacement > 1 m or absence; time-of-day; return probability; drop "half-life".
3. Zero-shot test meaningless at n ≈ 8 classes → time-dependent Brier over object-time pairs vs constant, 10 human guessers, PreSIST-Lang.
4. Ground-truth logistics contradictory; ethics approval missing → tags on top, masked in drone frames (+ ablation), ≥ 100 manual audits, ethics filed in month 0.
5. E2 unpowered/infeasible (800 flights) → counterfactual replay on E1 logs for the main table; ≥ 100 paired flights in the 1 h / 1 d cells; drop 1 min.
6. Strawman baselines → trust-then-search, verify-always, persistence filter, PreSIST prior, stationarity score, LLM-prior search.
7. Drone not justified → aerial verify (viewpoint + altitude per joule), energy in the cost, multi-height targets, fixed-height ablation.
8. Identical-chair re-ID → instance-exact vs intent-equivalent success; identical subset separately.
9. RQ3 is a solved task → grounding under stale memory, KnowNo conformal baseline, non-author command writers, risk–coverage with CIs.
10. Scope too large → cut zero-shot to one comparison, Phase 2 to a paragraph, LiDAR off the flight stack.

## Overclaims to remove
"the first per-class measurement" · "decay rates hand-set" · "update only when the robot sees change" · "people for Y seconds" · "both natural fits" (Jev) · "any outcome is publishable" · "calibrated confidence" (before it is measured).

## Closest prior work (verify all)
PreSIST https://arxiv.org/abs/2607.04057 · TPM (Toris & Chernova, ICRA 2017) · Persistence filter https://github.com/david-m-rosen/Persistence-Filter · Perpetua https://arxiv.org/abs/2507.18808 · PredictiveGraphs https://arxiv.org/abs/2605.00121 · Bogenberger et al. RA-L 2026 https://arxiv.org/abs/2509.19851 · FreMEn https://github.com/gestom/fremen/wiki · Bore et al. T-RO 2019 https://arxiv.org/pdf/1801.09292 · FlowMaps https://arxiv.org/abs/2606.20209 · Patel & Chernova CoRL 2022 · LT-Mem https://arxiv.org/abs/2608.19059 · VLMM https://arxiv.org/abs/2607.16173 · Memory for Attention https://arxiv.org/abs/2607.23797 · Quadrotor InstanceImageNav https://arxiv.org/abs/2606.29917 · KnowNo https://proceedings.mlr.press/v229/ren23a.html · ESC https://arxiv.org/abs/2301.13166 · Khronos https://github.com/MIT-SPARK/Khronos
