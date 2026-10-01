"""Run every simulated experiment and write results/RESULTS.md plus figures.

    python run_all.py            # full run (about a minute)
    python run_all.py --quick    # smoke run
"""
import argparse
import os
import time

import numpy as np

from lbyf import experiments as X
from lbyf import plots
from lbyf.config import SimConfig
from lbyf.stats import cluster_bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))


def ci(t, fmt="{:.3f}"):
    return f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"


def paired_time(e2, a, b, cfg):
    ta = np.array([r.time_s for r in e2["results"][a]])
    tb = np.array([r.time_s for r in e2["results"][b]])
    oid = np.array([c.oid for c in e2["cmds"]])
    return cluster_bootstrap(ta - tb, oid, cfg.n_boot, np.random.default_rng(cfg.seed + 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", choices=("open", "cubicle", "both"), default="both")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    layouts = ("open", "cubicle") if args.layout == "both" else (args.layout,)
    for layout in layouts:
        cfg = SimConfig.quick(seed=args.seed, layout=layout) if args.quick else SimConfig(seed=args.seed, layout=layout)
        run(cfg, os.path.join(args.out, layout))


def run(cfg, out):
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    ctx = X.setup(cfg)
    e1 = X.run_e1(ctx)
    e2 = X.run_e2(ctx)
    e2_rows = X.summarise_e2(e2, cfg)
    prim = X.primary_test(e2)
    budgets = np.array([10, 15, 20, 30, 45, 60, 90, 120, 240])
    curves = X.success_within_budget(e2, budgets)
    dr = X.decision_regions(ctx)
    e3 = X.run_e3(ctx)

    arm_names = [a.name for a in e2["arms"]]
    plots.floor_plan(ctx.world, ctx.drone, os.path.join(out, "fig0_floor_plan.png"))
    plots.persistence_curves(ctx, os.path.join(out, "fig1_persistence.png"))
    plots.decision_regions(dr, os.path.join(out, "fig2_decision_costs.png"))
    plots.success_vs_budget(curves, budgets, os.path.join(out, "fig3_success_vs_budget.png"), arm_names)
    plots.risk_coverage(e3, os.path.join(out, "fig4_risk_coverage.png"))

    L = []
    w = L.append
    w(f"# Simulation results: Look Before You Fly ({cfg.layout} layout)\n")
    w(f"Seed {cfg.seed} · {cfg.days} simulated days ({cfg.train_days} train / {cfg.days - cfg.train_days} test) · "
      f"{len(ctx.world.objects)} objects in {len(ctx.world.classes)} classes · {len(ctx.world.places)} places · "
      f"{ctx.drone.n_vp} viewpoints · runtime {time.time() - t_start:.0f} s.\n")
    w("All numbers come from the simulator in `sim/`, not from real flights. The LLM and human priors are "
      "documented stand-ins (hand-set commonsense medians), not real model outputs. CIs are 95 % "
      "object-cluster bootstrap.\n")

    w("## E1 · Which prior predicts \"still there\"? (lower Brier is better)\n")
    w(f"{e1['n_pairs']} held-out object-time pairs from {e1['n_objects']} objects, commands in working hours.\n")
    hz = [f"brier_{h:g}h" for h in cfg.horizons_h]
    w("| Prior | " + " | ".join(f"Brier @ {h:g} h" for h in cfg.horizons_h) + " | Brier, all |")
    w("|---|" + "---|" * (len(hz) + 1))
    for r in sorted(e1["table"], key=lambda r: r["brier_all"][0]):
        w(f"| {r['prior']} | " + " | ".join(ci(r[k]) for k in hz) + f" | {ci(r['brier_all'])} |")
    w("\nPer class (training move events, held-out empirical vs fitted P(still there)):\n")
    w("| Class | Train events | " + " | ".join(f"Empirical @ {h:g} h | Ours @ {h:g} h" for h in cfg.horizons_h) + " |")
    w("|---|---|" + "---|---|" * len(cfg.horizons_h))
    for r in e1["per_class"]:
        cells = " | ".join(f"{r[f'emp_{h:g}h']:.2f} | {r[f'ours_{h:g}h']:.2f}" for h in cfg.horizons_h)
        flag = "" if r["train_events"] >= 15 else " (pooled, < 15 events)"
        w(f"| {r['class']}{flag} | {r['train_events']} | {cells} |")

    w("\n## E2 · Command trials (paired, common random numbers)\n")
    w(f"{len(e2['cmds'])} commands ({cfg.n_commands_per_horizon} per horizon), "
      f"{sum(c.moved for c in e2['cmds'])} with the object moved since mapping. Budget {cfg.time_budget_s:.0f} s.\n")
    for subset in ["all"] + [f"{h:g}h" for h in cfg.horizons_h] + ["moved", "non-identical classes"]:
        rows = [r for r in e2_rows if r["subset"] == subset]
        if not rows:
            continue
        w(f"\n**Subset: {subset}** (n = {rows[0]['n']})\n")
        w("| Arm | Success (instance) | Success (intent) | Mean time (s) | Median time (s) | Energy (kJ) | Wasted looks | First actions |")
        w("|---|---|---|---|---|---|---|---|")
        for r in rows:
            acts = ", ".join(f"{k} {v}" for k, v in sorted(r["actions"].items()))
            w(f"| {r['arm']} | {ci(r['success'], '{:.2f}')} | {r['intent']:.2f} | {ci(r['time_mean'], '{:.1f}')} | "
              f"{r['time_median']:.1f} | {r['energy_kj']:.2f} | {r['wasted']:.2f} | {acts} |")

    w("\n### Pre-registered primary test\n")
    w(f"A5 Ours vs the better naive baseline ({prim['baseline']}) on instance success, exact McNemar: "
      f"{prim['rate_ours']:.3f} vs {prim['rate_baseline']:.3f}, discordant {prim['ours_only']} (ours only) / "
      f"{prim['baseline_only']} (baseline only), **p = {prim['p']:.3f}**.\n")
    w("Secondary (Holm-adjusted, instance success):\n")
    w("| Comparison | p | p (Holm) | First arm only | Second arm only |")
    w("|---|---|---|---|---|")
    for k, (p, a, b, adj) in prim["secondary"].items():
        w(f"| {k} | {p:.3f} | {adj:.3f} | {a} | {b} |")
    w("\nPaired mean time difference (first minus second, failures count as the full budget):\n")
    w("| Comparison | Δ time (s) |")
    w("|---|---|")
    for a, b in [("A5 Ours", "A1 Trust-then-search"), ("A5 Ours", "A2 Verify-always"),
                 ("A5 Ours", "A5-fixed Ours @0.4 m"), ("A1 Trust-then-search", "A1-fixed Trust-then-search @0.4 m"),
                 ("A5 Ours", "A3 Ours + persistence-filter prior")]:
        w(f"| {a} − {b} | {ci(paired_time(e2, a, b, cfg), '{:+.2f}')} |")

    w("\n### Success within a time budget\n")
    w("| Arm | " + " | ".join(f"{b:g} s" for b in budgets) + " |")
    w("|---|" + "---|" * len(budgets))
    for name in arm_names:
        w(f"| {name} | " + " | ".join(f"{v:.2f}" for v in curves[name]) + " |")

    w("\n### Decision analysis (laptop on a far desk, 1 day after mapping)\n")
    for key in ("full", "fixed"):
        ch = dr[key]["chosen"]
        ps = dr[key]["p"]
        segs, start = [], 0
        for i in range(1, len(ch) + 1):
            if i == len(ch) or ch[i] != ch[start]:
                segs.append(f"{ch[start]} for p in [{ps[start]:.2f}, {ps[i - 1]:.2f}]")
                start = i
        w(f"- {'All altitudes' if key == 'full' else 'Fixed 0.4 m'}: " + "; ".join(segs))

    w("\n## E3 · Grounding language references under stale memory\n")
    w(f"Conformal α = {cfg.conformal_alpha}; half the {cfg.n_grounding} commands calibrate, half test.\n")
    w("| Resolver | Argmax accuracy | Coverage | Ask rate | Ask rate when ambiguous | Accuracy when acting | Acted (unambiguous) |")
    w("|---|---|---|---|---|---|---|")
    for name, r in e3.items():
        s = r["summary"]
        w(f"| {name} | {s['argmax_accuracy']:.3f} | {s['coverage']:.3f} | {s['ask_rate']:.3f} | "
          f"{s['ask_rate_ambiguous']:.3f} | {s['accuracy_when_acting']:.3f} | {s['acted_fraction_unambiguous']:.3f} |")

    w("\n## Figures\n")
    for f, cap in [("fig0_floor_plan.png", "Simulated space"), ("fig1_persistence.png", "E1 persistence by class"),
                   ("fig2_decision_costs.png", "Expected cost of trust / verify / search vs P(still there)"),
                   ("fig3_success_vs_budget.png", "E2 success within a time budget"),
                   ("fig4_risk_coverage.png", "E3 risk vs coverage")]:
        w(f"![{cap}]({f})\n")

    with open(os.path.join(out, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
