# Hostile audit of proposal v4 (Expiry)

**Verdict as written:** IROS ≈ 25 %, RA-L ≈ 15 %, CoRL ≈ 5 %.
**If every v5 change is made:** IROS 45–50 %, RA-L 30–35 %, CoRL ≈ 10 % (no learning contribution).
**Reviewer summary line:** "Careful but incremental benchmark. Dense oracle replay is NoScope/Reducto-era frame filtering applied to a controller. Open-loop harm replay on handheld footage is not a valid counterfactual for a flying drone."

## v3 kill shots after v4
| v3 issue | Status | Why |
|---|---|---|
| K1 false novelty | Partial | "Nobody has measured it", "first", "exact" still attackable |
| K2 circular harm | Partial | Δ clearance is geometric; only the person↔mannequin stratum is non-geometric |
| K3 empty certificate | Fixed | Dropped (delete the "secondary bound" too) |
| Heuristic wins | Fixed by reframing | Needs a headline that survives it |
| Rigged by CBF | Not fixed | AlphaAdj already has stale-request mitigation; harm may be ≈ 0 |
| ≡ ill-defined | Partial | ε, H unspecified; equivalence on a trajectory the controller did not generate |
| Go/no-go | Papered over | Non-numeric |

## New kill shots
| # | Issue | Fix |
|---|---|---|
| N1 | Open-loop counterfactual invalid: handheld path ≠ controller's path; viewpoint, speed and vibration differ; "exact" is false | Branching SITL/point-mass rollouts (H ≤ 3 s) in a ZED-reconstructed scene with replayed human tracks; calibrate predicted vs measured clearance on flights; say "open-loop-exact" |
| N2 | VLM latency ignored (oracle decisions are instant) | Latency + queue model with Orin-measured distributions; staleness = trigger delay + queue + inference |
| N3 | Dense reference replay is NoScope (VLDB'17) / Reducto (SIGCOMM'20) | Cite them + Chameleon + FilterForward; claim only the differences: controller parameter as query, physical cost, VLM noise as a cause, semantic changes with small pixel delta |

## Major flaws → fixes
1. Host fidelity (student re-implementations; ASMA ≠ AlphaAdj mechanism; EAMP has its own trigger) → use ASMA code, verbatim prompts, PC-SET as an arm, 3 prompt paraphrases per host.
2. Taxonomy on natural footage unidentified; interactions ignored; sampling noise ≠ perceptual jitter → 2×2 factorial (image × geometry) Shapley attribution, 4 causes, report the interaction share.
3. Staged + natural data mixed → natural only for rates; staged only for attribution and stress tests.
4. Statistics (≈ 240 tests; ±10 pp CIs at ~100 events) → one pre-registered primary test, Holm for the rest, episode-cluster bootstrap, recall at fixed rates {0.1, 0.25, 0.5, 1, 2} Hz, recall-AUC over log rate, per-episode p95 rate.
5. "Comfort distance" cherry-picked → "proxemic intrusion" swept 0.6–2.0 m with sourced numbers; never say "comfort" without participants.
6. Flights underpowered for rank transfer → ≈ 72 paired runs; claim replay-model validity (R², bias), not rank transfer.
7. Host-1 harm may be ≈ 0 → pre-register it as acceptable; report the efficiency cost.
8. VLM families not independent (Molmo is built on Qwen2, verify) → e.g. Qwen2.5-VL-7B + InternVL3/Gemma-3.
9. Feasibility/go-no-go → month 1 = one host + pilot; numeric gates.

## Headline to aim for
Kaplan–Meier **decision half-life** curves per host × VLM, split by cause. "X % of flips are noise, Y % behaviourally irrelevant; of the Z % that matter, a geometric heuristic catches W % at 0.5 Hz; the semantic-only residual is where learned gates earn their compute."

## Jev
Move it to an appendix or the workshop by-product; pin its version and log all I/O; keep it out of the main reproducible ranking.

Sources: NoScope https://www.vldb.org/pvldb/vol10/p1586-kang.pdf · Reducto https://web.cs.ucla.edu/~harryxu/papers/li-sigcomm20.pdf · Chameleon https://dl.acm.org/doi/10.1145/3230543.3230574 · FilterForward https://ar5iv.labs.arxiv.org/html/1905.13536 · AlphaAdj https://arxiv.org/abs/2603.21142 · ASMA https://arxiv.org/abs/2409.10283 · EAMP https://arxiv.org/abs/2606.25629 · Slow Brain Fast Planner https://arxiv.org/abs/2606.20458 · ProAct-VLM https://arxiv.org/html/2609.37681 · HDI distance https://pmc.ncbi.nlm.nih.gov/articles/PMC11503297/ (verify all)
