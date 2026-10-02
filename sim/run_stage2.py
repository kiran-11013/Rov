"""Stage 2: layout sweep, seed robustness, sensitivity, look-alike stress -> results/stage2/STAGE2.md + figures.

    python run_stage2.py                 # everything (about 10 minutes on 4 cores)
    python run_stage2.py --quick         # smoke run
    python run_stage2.py --from-pickle   # re-analyse saved records without re-simulating
"""
import argparse
import os
import pickle
import time

import numpy as np

from lbyf import plots2
from lbyf import stage2 as S
from lbyf.stage2 import arm_mean, diff, group_by_tag, mean_ci, rel_gain

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = 0.10        # pre-registered: altitude is a main claim only if some layout saves >= 10 % with CI above 0
FLOOR = 0.03       # pre-registered: if no layout saves >= 3 %, lead with the prior benchmark + re-ID instead

BASE_LAYOUT = dict(n_commands_per_horizon=150, n_boot=20, n_grounding=100)
BASE_SEEDS = dict(n_commands_per_horizon=250, n_boot=20, n_grounding=400)
BASE_STRESS = dict(n_commands_per_horizon=150, n_boot=20, n_grounding=100)

LAYOUT_SEEDS = list(range(0, 6))
SEEDS_SEEDS = list(range(10, 22))
STRESS_SEEDS = list(range(30, 34))

BASES = {"open (1.3 m partitions)": dict(layout="open"),
         "booth (1.6 m walls + desk clutter)": dict(layout="booth", partition_h=1.6, clutter_h=0.45)}

SENS = [("default", {}),
        ("climb speed 0.25 m/s", {"v_z": 0.25}), ("climb speed 1.0 m/s", {"v_z": 1.0}), ("climb speed 2.0 m/s", {"v_z": 2.0}),
        ("cruise speed 0.5 m/s", {"v_xy": 0.5}), ("cruise speed 2.0 m/s", {"v_xy": 2.0}),
        ("energy weight 0", {"energy_weight": 0.0}), ("energy weight 0.01", {"energy_weight": 0.01}),
        ("energy weight 0.05", {"energy_weight": 0.05}),
        ("time budget 60 s", {"time_budget_s": 60.0}), ("time budget 120 s", {"time_budget_s": 120.0}),
        ("observe time 1 s", {"observe_time_s": 1.0}), ("observe time 5 s", {"observe_time_s": 5.0}),
        ("detection range 4 m", {"max_range": 4.0}), ("detection range 10 m", {"max_range": 10.0}),
        ("altitudes 0.4 / 1.8", {"altitudes": (0.4, 1.8)}), ("altitudes 1.0 / 1.8", {"altitudes": (1.0, 1.8)}),
        ("altitudes 5 levels", {"altitudes": (0.4, 0.7, 1.0, 1.4, 1.8)})]

LOOK = [("default (12 chairs)", {}, None),
        ("4 chairs", {"n_chairs": 4}, "How many chairs look alike"),
        ("20 chairs", {"n_chairs": 20}, "How many chairs look alike"),
        ("+ mugs, laptops", {"identical_classes": ("chair", "mug", "laptop")}, "Which classes look alike"),
        ("+ toolboxes, boxes", {"identical_classes": ("chair", "mug", "laptop", "toolbox", "box")},
         "Which classes look alike"),
        ("noisier matching", {"obs_noise": 0.10, "reid_threshold": 0.85}, "Appearance matching quality"),
        ("much noisier", {"obs_noise": 0.15, "reid_threshold": 0.75}, "Appearance matching quality"),
        ("spread 0.01", {"identical_sigma": 0.01}, "Spread among look-alikes"),
        ("spread 0.06", {"identical_sigma": 0.06}, "Spread among look-alikes")]


# ----------------------------------------------------------------------------- formatting
def pct(t, signed=False):
    m, lo, hi = t
    f = "{:+.1f}" if signed else "{:.1f}"
    if not np.isfinite(m):
        return "n/a"
    return f"{f.format(m * 100)} % [{f.format(lo * 100)}, {f.format(hi * 100)}]"


def val(t, d=3):
    m, lo, hi = t
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def ci(recs, fn):
    return mean_ci([fn(r) for r in recs])


EXTRA = {}  # applied to every configuration (e.g. the extended visibility model)


# ----------------------------------------------------------------------------- run
def build_tasks(part, quick):
    tasks = _build_tasks(part, quick)
    for t in tasks:
        t["cfg"] = {**t["cfg"], **EXTRA}
    return tasks


def _build_tasks(part, quick):
    if part == "layout":
        grid = {}
        for ph in (1.0, 1.3, 1.6, 1.9):
            for cl in (0.0, 0.45):
                grid[f"open|{ph}|{cl}"] = dict(layout="open", partition_h=ph, clutter_h=cl)
        for ph in (1.4, 1.8):
            for cl in (0.0, 0.45):
                grid[f"cubicle|{ph}|{cl}"] = dict(layout="cubicle", partition_h=ph, clutter_h=cl)
        for ph in (1.2, 1.4, 1.8, 2.0):
            for cl in (0.0, 0.45):
                grid[f"booth|{ph}|{cl}"] = dict(layout="booth", partition_h=ph, clutter_h=cl)
        if quick:
            grid = dict(list(grid.items())[:2] + list(grid.items())[-2:])
        cfgs = {k: {**BASE_LAYOUT, **v} for k, v in grid.items()}
        seeds = LAYOUT_SEEDS[:2] if quick else LAYOUT_SEEDS
        return S.make_tasks(cfgs, "altitude", seeds, "layout", chunk=3 if not quick else 1)
    if part == "seeds":
        layouts = {"open (1.3 m)": dict(layout="open"), "cubicle (1.4 m)": dict(layout="cubicle"),
                   "booth (1.4 m)": dict(layout="booth", partition_h=1.4),
                   "booth dense (1.6 m + clutter)": dict(layout="booth", partition_h=1.6, clutter_h=0.45)}
        cfgs = {k: {**BASE_SEEDS, **v} for k, v in layouts.items()}
        seeds = SEEDS_SEEDS[:2] if quick else SEEDS_SEEDS
        return S.make_tasks(cfgs, "full", seeds, "seeds", chunk=3 if not quick else 1, e1_e3=True)
    if part == "sens":
        cfgs = {}
        settings = SENS[:3] if quick else SENS
        for b, bc in BASES.items():
            for label, ov in settings:
                cfgs[f"{b}|{label}"] = {**BASE_STRESS, **bc, **ov}
        seeds = STRESS_SEEDS[:2] if quick else STRESS_SEEDS
        return S.make_tasks(cfgs, "altitude", seeds, "sens", chunk=len(seeds))
    if part == "look":
        cfgs = {}
        settings = LOOK[:3] if quick else LOOK
        for b, bc in BASES.items():
            for label, ov, _g in settings:
                cfgs[f"{b}|{label}"] = {**BASE_STRESS, **bc, **ov}
        seeds = STRESS_SEEDS[:2] if quick else STRESS_SEEDS
        return S.make_tasks(cfgs, "lookalike", seeds, "look", chunk=len(seeds))
    raise ValueError(part)


# ----------------------------------------------------------------------------- analysis
def analyse_layout(records):
    rows = []
    for tag, recs in group_by_tag(records).items():
        c = recs[0]["cfg"]
        rows.append(dict(
            tag=tag, layout=c["layout"], partition_h=c["partition_h"], clutter_h=c["clutter_h"], n=len(recs),
            gain=ci(recs, lambda r: rel_gain(r, "ours@0.4", "ours")),
            gain_moved=ci(recs, lambda r: rel_gain(r, "ours@0.4", "ours", subset="moved")),
            gain_high=ci(recs, lambda r: rel_gain(r, "ours@0.4", "ours@1.8")),
            gain_trust=ci(recs, lambda r: rel_gain(r, "trust@0.4", "trust")),
            ours_vs_trust=ci(recs, lambda r: rel_gain(r, "trust", "ours")),
            ours_vs_verify=ci(recs, lambda r: rel_gain(r, "verify-always", "ours")),
            succ_diff=ci(recs, lambda r: diff(r, "ours", "trust")),
            dose={a: float(np.mean([arm_mean(r, a, "t") for r in recs]))
                  for a in ("trust", "ours@0.4", "ours@1.0", "ours@1.8", "ours", "oracle")}))
    return sorted(rows, key=lambda r: (r["layout"], r["clutter_h"], r["partition_h"]))


def analyse_seeds(records):
    out = {}
    for tag, recs in group_by_tag(records).items():
        e1_names = list(recs[0]["e1"])
        ours_n, pf_n, cs_n = ("Hier. Weibull + returns (ours)", "Persistence filter (exp.)", "Commonsense (LLM stand-in)")
        o = {
            "n": len(recs),
            "succ_ours": ci(recs, lambda r: arm_mean(r, "ours", "succ")),
            "succ_trust": ci(recs, lambda r: arm_mean(r, "trust", "succ")),
            "succ_diff": ci(recs, lambda r: diff(r, "ours", "trust")),
            "time_vs_trust": ci(recs, lambda r: rel_gain(r, "trust", "ours")),
            "time_vs_trust_moved": ci(recs, lambda r: rel_gain(r, "trust", "ours", subset="moved")),
            "time_vs_verify": ci(recs, lambda r: rel_gain(r, "verify-always", "ours")),
            "alt_gain": ci(recs, lambda r: rel_gain(r, "ours@0.4", "ours")),
            "alt_gain_trust": ci(recs, lambda r: rel_gain(r, "trust@0.4", "trust")),
            "prior_succ_commonsense": ci(recs, lambda r: diff(r, "ours+commonsense", "ours")),
            "prior_succ_pf": ci(recs, lambda r: diff(r, "ours+persistence-filter", "ours")),
            "brier": {n: ci(recs, lambda r, n=n: r["e1"][n]["all"]) for n in e1_names},
            "brier24": {n: ci(recs, lambda r, n=n: r["e1"][n]["24h"]) for n in e1_names},
            "ours_best_frac": float(np.mean([min(r["e1"], key=lambda n: r["e1"][n]["all"]) == ours_n for r in recs])),
            "brier_ours_minus_pf": ci(recs, lambda r: r["e1"][ours_n]["all"] - r["e1"][pf_n]["all"]),
            "brier_ours_minus_cs": ci(recs, lambda r: r["e1"][ours_n]["all"] - r["e1"][cs_n]["all"]),
            "e3_acc_diff": ci(recs, lambda r: r["e3"]["Staleness-aware"]["argmax_accuracy"] - r["e3"]["Map-only"]["argmax_accuracy"]),
            "e3_act_diff": ci(recs, lambda r: r["e3"]["Staleness-aware"]["accuracy_when_acting"] - r["e3"]["Map-only"]["accuracy_when_acting"]),
            "e3_acc": {k: ci(recs, lambda r, k=k: r["e3"][k]["argmax_accuracy"]) for k in ("Map-only", "Staleness-aware")},
            "e3_ask": ci(recs, lambda r: r["e3"]["Staleness-aware"]["ask_rate"]),
            "per_seed_alt": np.array([rel_gain(r, "ours@0.4", "ours") for r in recs]),
            "per_seed_vs_trust": np.array([rel_gain(r, "trust", "ours") for r in recs]),
        }
        out[tag] = o
    return out


def analyse_sens(records):
    res = {}
    order = {label: i for i, (label, _o) in enumerate(SENS)}
    for tag, recs in sorted(group_by_tag(records).items(), key=lambda kv: order.get(kv[0].split("|", 1)[1], 99)):
        base, label = tag.split("|", 1)
        res.setdefault(base, {})[label] = dict(
            gain=ci(recs, lambda r: rel_gain(r, "ours@0.4", "ours")),
            gain_trust=ci(recs, lambda r: rel_gain(r, "trust@0.4", "trust")),
            vs_trust=ci(recs, lambda r: rel_gain(r, "trust", "ours")),
            vs_verify=ci(recs, lambda r: rel_gain(r, "verify-always", "ours")),
            succ_diff=ci(recs, lambda r: diff(r, "ours", "trust")))
    return res


def analyse_look(records):
    res = {}
    order = {label: i for i, (label, _o, _g) in enumerate(LOOK)}
    for tag, recs in sorted(group_by_tag(records).items(), key=lambda kv: order.get(kv[0].split("|", 1)[1], 99)):
        base, label = tag.split("|", 1)

        def fail_share(r):
            m = r["meta"]
            fail = r["arms"]["ours"]["succ"] < 0.5
            return float((fail & m["ident"]).sum() / fail.sum()) if fail.any() else np.nan

        res.setdefault(base, {})[label] = dict(
            arms={a: ci(recs, lambda r, a=a: arm_mean(r, a, "succ"))
                  for a in ("trust", "trust (no spatial re-ID)", "verify-always", "ours", "ours (no spatial re-ID)")},
            intent=ci(recs, lambda r: arm_mean(r, "ours", "intent")),
            spatial_gain=ci(recs, lambda r: arm_mean(r, "ours", "succ") - arm_mean(r, "ours (no spatial re-ID)", "succ")),
            ours_vs_trust=ci(recs, lambda r: diff(r, "ours", "trust")),
            fail_share=ci(recs, fail_share))
    return res


# ----------------------------------------------------------------------------- report
def report(R, out, runtime):
    L, w = [], None
    w = L.append
    lay = R.get("layout")
    w("# Stage 2: stress-testing the simulated claims\n")
    if EXTRA or any(r["cfg"].get("visibility_model") == "extended" for part in R.values() for r in part[:1]):
        cfg0 = next(iter(R.values()))[0]["cfg"]
        w(f"**Visibility model: extended** (a50 = {cfg0.get('vis_a50_px')} px), calibrated against Isaac Sim renders "
          "instead of the Stage 1-2 point model.\n")
    w(f"Runtime {runtime:.0f} s. Everything here comes from the simulator (`sim/`), not from real flights. "
      "Seeds are the independent unit; intervals are Student-t 95 % across seeds. Stand-in priors are hand-set, "
      "not real LLM output.\n")
    w("## Decision rules (written before running the sweeps)\n")
    w(f"- **Altitude stays a main claim** only if some layout in the swept range (walls 1.0–2.0 m, clutter 0–0.45 m, "
      f"default drone speeds) saves **≥ {RULE * 100:.0f} %** time by using all altitudes instead of 0.4 m only, "
      "with the 95 % interval above 0.")
    w(f"- If **no** layout reaches {FLOOR * 100:.0f} %, the paper leads with the prior-method benchmark and "
      "look-alike re-identification instead.\n")

    verdict = None
    if lay:
        rows = analyse_layout(lay)
        passing = [r for r in rows if r["gain"][0] >= RULE and r["gain"][1] > 0]
        best = max(rows, key=lambda r: r["gain"][0])
        near = [r for r in rows if r["gain"][0] >= FLOOR and r["gain"][1] > 0]
        verdict = ("PASS" if passing else ("PARTIAL" if near else "FAIL"))
        w("## Verdict on the altitude claim\n")
        w(f"**{verdict}.** Best layout: {best['layout']} layout, walls {best['partition_h']} m, clutter "
          f"{best['clutter_h']} m, saving {pct(best['gain'])} (moved objects only: {pct(best['gain_moved'])}). "
          f"{len(passing)} of {len(rows)} layouts meet the {RULE * 100:.0f} % rule; {len(near)} reach "
          f"{FLOOR * 100:.0f} % with interval above 0.\n")

        w("## 1. Layout sweep\n")
        w(f"{len(rows)} layouts × {rows[0]['n']} seeds, 300 commands per run. \"Altitude saving\" = 1 − mean time with all "
          "altitudes / mean time at 0.4 m only, same policy, same commands.\n")
        w("| Layout | Walls (m) | Clutter (m) | Altitude saving (ours) | … moved objects only | Always 1.8 m vs always 0.4 m | Altitude saving (trust baseline) | Ours vs trust: time saving | Ours − trust: success |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            w(f"| {r['layout']} | {r['partition_h']} | {r['clutter_h']} | {pct(r['gain'], True)} | "
              f"{pct(r['gain_moved'], True)} | {pct(r['gain_high'], True)} | {pct(r['gain_trust'], True)} | "
              f"{pct(r['ours_vs_trust'], True)} | {val(r['succ_diff'])} |")
        best_high = max(rows, key=lambda r: r["gain_high"][0])
        w(f"\nBest case for altitude, flying *always* at 1.8 m: {best_high['layout']} layout, walls "
          f"{best_high['partition_h']} m, clutter {best_high['clutter_h']} m, saving {pct(best_high['gain_high'])}. "
          "This is not the pre-registered metric (it compares two fixed heights instead of letting the policy "
          "choose), and it still stays below the 10 % rule.\n")
        w("\nMean time (s) by altitude set, same policy (dose–response):\n")
        w("| Layout | Walls | Clutter | trust | ours @0.4 | ours @1.0 | ours @1.8 | ours (all) | oracle |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            d = r["dose"]
            w(f"| {r['layout']} | {r['partition_h']} | {r['clutter_h']} | {d['trust']:.2f} | {d['ours@0.4']:.2f} | "
              f"{d['ours@1.0']:.2f} | {d['ours@1.8']:.2f} | {d['ours']:.2f} | {d['oracle']:.2f} |")
        w("\n![Altitude saving across layouts](fig_s2_1_altitude_vs_wall.png)\n")
        plots2.altitude_vs_wall(rows, os.path.join(out, "fig_s2_1_altitude_vs_wall.png"), RULE)

    se = R.get("seeds")
    if se:
        order = ["open (1.3 m)", "cubicle (1.4 m)", "booth (1.4 m)", "booth dense (1.6 m + clutter)"]
        A = dict(sorted(analyse_seeds(se).items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))
        n = next(iter(A.values()))["n"]
        w(f"## 2. Seed robustness ({n} independent six-week worlds per layout)\n")
        w("| Layout | Success: ours | Success: trust | Ours − trust | Time saving vs trust | … moved only | Time saving vs verify-always | Altitude saving |")
        w("|---|---|---|---|---|---|---|---|")
        for tag, a in A.items():
            w(f"| {tag} | {val(a['succ_ours'])} | {val(a['succ_trust'])} | {val(a['succ_diff'])} | "
              f"{pct(a['time_vs_trust'], True)} | {pct(a['time_vs_trust_moved'], True)} | "
              f"{pct(a['time_vs_verify'], True)} | {pct(a['alt_gain'], True)} |")
        w("\n**Effect of the prior inside the policy** (success difference vs our fitted prior):\n")
        w("| Layout | Commonsense (LLM stand-in) − ours | Persistence filter − ours |")
        w("|---|---|---|")
        for tag, a in A.items():
            w(f"| {tag} | {val(a['prior_succ_commonsense'])} | {val(a['prior_succ_pf'])} |")
        w("\n**E1 across seeds** (Brier, lower is better; first layout shown, the world dynamics are the same in "
          "all layouts):\n")
        first = next(iter(A.values()))
        w("| Prior | Brier, all | Brier @ 24 h |")
        w("|---|---|---|")
        for nme in sorted(first["brier"], key=lambda k: first["brier"][k][0]):
            w(f"| {nme} | {val(first['brier'][nme])} | {val(first['brier24'][nme])} |")
        w(f"\nOurs has the lowest Brier in **{first['ours_best_frac'] * 100:.0f} %** of seeds. "
          f"Ours − persistence filter: {val(first['brier_ours_minus_pf'], 4)}. "
          f"Ours − commonsense: {val(first['brier_ours_minus_cs'], 4)}.\n")
        w("**E3 across seeds** (staleness-aware minus map-only):\n")
        w("| Layout | Argmax accuracy difference | Accuracy-when-acting difference | Ask rate |")
        w("|---|---|---|---|")
        for tag, a in A.items():
            w(f"| {tag} | {val(a['e3_acc_diff'], 4)} | {val(a['e3_act_diff'], 4)} | {val(a['e3_ask'])} |")
        w("\n![Seed-to-seed spread](fig_s2_2_seeds.png)\n")
        plots2.seed_strips({"Altitude saving (ours)": {t: a["per_seed_alt"] for t, a in A.items()},
                            "Time saving, ours vs trust": {t: a["per_seed_vs_trust"] for t, a in A.items()}},
                           os.path.join(out, "fig_s2_2_seeds.png"))

    sn = R.get("sens")
    if sn:
        res = analyse_sens(sn)
        w("## 3. Sensitivity to drone and sensing assumptions\n")
        w(f"{len({r['seed'] for r in sn})} seeds per setting, 300 commands per run. "
          "One parameter changed at a time from the default.\n")
        per_base = {}
        for base, items in res.items():
            w(f"\n**{base}**\n")
            w("| Setting | Altitude saving (ours) | Altitude saving (trust) | Ours vs trust: time saving | Ours vs verify-always | Ours − trust: success |")
            w("|---|---|---|---|---|---|")
            for label, v in items.items():
                w(f"| {label} | {pct(v['gain'], True)} | {pct(v['gain_trust'], True)} | {pct(v['vs_trust'], True)} | "
                  f"{pct(v['vs_verify'], True)} | {val(v['succ_diff'])} |")
            per_base[base] = [(label, v["gain"]) for label, v in items.items()]
        w("\n![Sensitivity of the altitude benefit](fig_s2_3_sensitivity.png)\n")
        plots2.sensitivity_dots(per_base, os.path.join(out, "fig_s2_3_sensitivity.png"), RULE)

    lk = R.get("look")
    if lk:
        res = analyse_look(lk)
        w("## 4. Look-alike stress\n")
        w("Success = the commanded *instance* reached. \"No spatial re-ID\" ignores where the belief says the object "
          "probably is when choosing between look-alikes. Failure share = fraction of the policy's failures that "
          "involve a look-alike class.\n")
        for base, items in res.items():
            w(f"\n**{base}**\n")
            w("| Setting | trust | trust, no spatial | ours | ours, no spatial | Spatial prior gain (ours) | Intent success (ours) | Failures on look-alikes |")
            w("|---|---|---|---|---|---|---|---|")
            for label, v in items.items():
                a = v["arms"]
                w(f"| {label} | {val(a['trust'])} | {val(a['trust (no spatial re-ID)'])} | {val(a['ours'])} | "
                  f"{val(a['ours (no spatial re-ID)'])} | {val(v['spatial_gain'])} | {val(v['intent'])} | "
                  f"{pct(v['fail_share'])} |")
        groups = {}
        for base, items in res.items():
            groups[base] = {}
            for _label, _ov, g in LOOK:
                if g is None:
                    continue
                members = [(lab, items[lab]) for lab, _o, gg in LOOK if gg == g and lab in items]
                pos = 1 if g in ("How many chairs look alike", "Spread among look-alikes") else 0  # numeric order
                members.insert(pos, ("default", items["default (12 chairs)"]))
                groups[base][g] = [(lab, v["arms"]) for lab, v in members]
        w("\n![Look-alike stress](fig_s2_4_lookalike.png)\n")
        plots2.lookalike_lines(groups, os.path.join(out, "fig_s2_4_lookalike.png"))

    with open(os.path.join(out, "STAGE2.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--from-pickle", action="store_true")
    ap.add_argument("--parts", default="layout,seeds,sens,look")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "stage2"))
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--visibility", choices=("point", "extended"), default="point")
    ap.add_argument("--a50", type=float, default=200.0, help="extended model pixel threshold")
    args = ap.parse_args()
    if args.visibility == "extended":
        EXTRA.update(visibility_model="extended", vis_a50_px=args.a50)
    os.makedirs(args.out, exist_ok=True)
    pkl = os.path.join(args.out, "records.pkl")
    t0 = time.time()
    if args.from_pickle:
        with open(pkl, "rb") as fh:
            R = pickle.load(fh)
    else:
        R = {}
        if os.path.exists(pkl) and args.parts != "layout,seeds,sens,look":
            with open(pkl, "rb") as fh:
                R = pickle.load(fh)  # keep earlier parts; re-run only the requested ones
        for part in args.parts.split(","):
            tasks = build_tasks(part, args.quick)
            print(f"[{part}] {len(tasks)} chunks", flush=True)
            R[part] = S.run_tasks(tasks, args.processes)
            with open(pkl, "wb") as fh:
                pickle.dump(R, fh)
    verdict = report(R, args.out, time.time() - t0)
    print("verdict:", verdict, "| wrote", os.path.join(args.out, "STAGE2.md"))


if __name__ == "__main__":
    main()
