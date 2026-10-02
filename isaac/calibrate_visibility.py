"""Calibrate the extended visibility model on Isaac Sim renders, with leave-one-layout-out validation.

Inputs are the render index files written by capture.py. Scene specs are regenerated here with the same defaults
used on the render machine (seed 0, 30 viewpoint nodes, default time), and checked against the index.

    python isaac/calibrate_visibility.py --render booth=isaac/results/renders/booth_index.json \
        --render open=isaac/results/renders/open_index.json --render cubicle=isaac/results/renders/cubicle_index.json

Model variants (an ablation ladder):
    point     the Stage 1-2 model (one point 0.2 m above the support, range fall-off)
    extent    object size and height: 3 points up the object, pixel area from real dimensions
    +vfov     ... plus the camera's vertical field of view
    +top      ... plus the top surface seen from above (full extended model)
For each variant the pixel threshold a50 is fitted on two layouts and scored on the held-out third.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)

import analyze_visibility as AV  # noqa: E402
import export_scene as ES  # noqa: E402
from lbyf.config import SimConfig  # noqa: E402
from lbyf.visibility import VisParams, footprint, visible_pixels  # noqa: E402
from lbyf.world import make_world  # noqa: E402

VARIANTS = {"extent": dict(use_vfov=False, use_top=False), "+vfov": dict(use_vfov=True, use_top=False),
            "+top": dict(use_vfov=True, use_top=True)}
A50_GRID = np.round(np.geomspace(25, 3200, 43), 1)
MIN_RENDER_PX = 200


def load_layout(name, index_path):
    cfg = SimConfig(layout=name)
    spec = ES.build_spec(cfg)
    with open(index_path) as fh:
        index = json.load(fh)
    keys = {(v["node"], round(v["alt"], 3)) for v in spec["viewpoints"]}
    seen = {(f["node"], round(f["alt"], 3)) for f in index["frames"]}
    if not seen <= keys:
        raise SystemExit(f"{name}: render viewpoints do not match the regenerated spec - was capture run with the "
                         f"default export settings (seed 0, 30 nodes, default time)?")
    oids = {str(o["oid"]) for o in spec["objects"]}
    extra = {k for f in index["frames"] for k in f["pixels"]} - oids
    if extra:
        raise SystemExit(f"{name}: render contains objects not in the spec: {sorted(extra)[:5]}")
    world = make_world(cfg.seed, **cfg.world_kwargs())
    return {"name": name, "cfg": cfg, "spec": spec, "index": index, "world": world}


def pair_table(L, prm):
    """For every rendered (viewpoint, object): render px (max over yaws), model px, point-model p, alt, cls."""
    render = {}
    for f in L["index"]["frames"]:
        for oid, px in f["pixels"].items():
            k = (f["node"], round(f["alt"], 3), int(oid))
            render[k] = max(render.get(k, 0), int(px))
    rendered = {(f["node"], round(f["alt"], 3)) for f in L["index"]["frames"]}
    rows = []
    for vp in L["spec"]["viewpoints"]:
        key = (vp["node"], round(vp["alt"], 3))
        if key not in rendered:
            continue
        eye = (vp["x"], vp["y"], vp["alt"])
        for ob in L["spec"]["objects"]:
            w, h, top = footprint(ob["shape"], ob["dims"])
            mpx = visible_pixels(L["world"], eye, ob["x"], ob["y"], ob["z_base"], w, h, top, prm) if prm else 0.0
            rows.append({"node": vp["node"], "alt": vp["alt"], "oid": ob["oid"], "cls": ob["cls"],
                         "render_px": render.get(key + (ob["oid"],), 0), "model_px": mpx,
                         "point_p": vp["model_p"][str(ob["oid"])]})
    return rows


def score(rows, model_vis):
    r = np.array([x["render_px"] >= MIN_RENDER_PX for x in rows])
    m = np.asarray(model_vis, bool)
    alts = np.array([x["alt"] for x in rows])
    nodes = {a: len({x["node"] for x in rows if x["alt"] == a}) for a in set(alts)}
    lo, hi = min(nodes), max(nodes)
    per_vp = lambda v, a: v[alts == a].sum() / nodes[a]  # noqa: E731
    return {"agreement": float(np.mean(m == r)), "kappa": AV.kappa(m, r),
            "model_gain": per_vp(m, hi) / max(per_vp(m, lo), 1e-9) - 1,
            "render_gain": per_vp(r, hi) / max(per_vp(r, lo), 1e-9) - 1,
            "model_per_vp": {a: per_vp(m, a) for a in sorted(nodes)},
            "render_per_vp": {a: per_vp(r, a) for a in sorted(nodes)}}


def threshold(rows, a50):
    return [x["model_px"] >= a50 for x in rows]


def fit_a50(train_rows):
    best = max(A50_GRID, key=lambda a: (score(train_rows, threshold(train_rows, a))["agreement"], -abs(np.log(a / 200))))
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="append", required=True, help="layout=path/to/index.json (repeat)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "CALIBRATION.md"))
    args = ap.parse_args()
    layouts = [load_layout(*r.split("=", 1)) for r in args.render]
    names = [L["name"] for L in layouts]

    tables = {"point": {L["name"]: pair_table(L, None) for L in layouts}}
    for v, opts in VARIANTS.items():
        prm = VisParams(**opts)
        tables[v] = {L["name"]: pair_table(L, prm) for L in layouts}

    lines = ["# Stage 3a: calibrating the visibility model on Isaac Sim renders\n",
             f"Layouts: {', '.join(names)}. Render-visible = ≥ {MIN_RENDER_PX} px in any yaw. For each extended variant, "
             "the pixel threshold a50 is fitted on the other layouts and scored on the held-out one "
             "(leave-one-layout-out). \"Gain\" = objects visible per viewpoint at the highest altitude vs the lowest.\n",
             "| Variant | Held-out layout | a50 (px, fitted on others) | Agreement | κ | Model gain 0.4→1.8 m | Render gain 0.4→1.8 m |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for v in ["point"] + list(VARIANTS):
        accs = []
        for held in names:
            rows = tables[v][held]
            if v == "point":
                a50, vis = None, [x["point_p"] >= 0.5 for x in rows]
            else:
                train = [x for n in names if n != held for x in tables[v][n]] or rows
                a50 = fit_a50(train)
                vis = threshold(rows, a50)
            s = score(rows, vis)
            accs.append(s["agreement"])
            lines.append(f"| {v} | {held} | {'—' if a50 is None else f'{a50:g}'} | {s['agreement']:.1%} | {s['kappa']:.2f} | "
                         f"{s['model_gain']:+.0%} | {s['render_gain']:+.0%} |")
        summary[v] = float(np.mean(accs))

    best = max(VARIANTS, key=lambda v: summary[v])
    all_rows = [x for n in names for x in tables[best][n]]
    a50_all = fit_a50(all_rows)
    s_all = score(all_rows, threshold(all_rows, a50_all))
    lines += ["", "**Mean held-out agreement:** " + ", ".join(f"{v} {summary[v]:.1%}" for v in summary) + ".\n",
              f"**Chosen:** `{best}` with a50 = **{a50_all:g} px** fitted on all layouts "
              f"(agreement {s_all['agreement']:.1%}, κ {s_all['kappa']:.2f}). Objects visible per viewpoint by altitude:\n",
              "| Altitude | Model (calibrated) | Render |", "|---|---|---|"]
    for a in s_all["model_per_vp"]:
        lines.append(f"| {a:g} m | {s_all['model_per_vp'][a]:.1f} | {s_all['render_per_vp'][a]:.1f} |")
    lines.append(f"\nUse in the simulator: `SimConfig(visibility_model=\"extended\", vis_a50_px={a50_all:g})`.\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines))
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump({"chosen": best, "a50_px": a50_all, "heldout_agreement": summary}, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
