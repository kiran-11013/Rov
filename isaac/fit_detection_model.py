"""Fit the simulator's detection model to real-detector results (YOLO-World on Isaac Sim renders).

Input: detection_rows.json from detect_frames.py for one or more layouts (default seed / time, as exported by
export_scene.py). For every (viewpoint, object) pair we recompute what the *simulator* predicts (visible pixels of
the proxy, extended model) and the viewing angle, then fit, per size group (tall / medium / flat / small, as in
sim/lbyf/visibility.py GROUP_OF), a logistic model

    p(real detection) = sigmoid(b0 + b1 * ln(sim pixels) + b2 * down / 30deg)

where down = how steeply the camera looks down at the object (degrees, 0 = level). Pairs the simulator calls
invisible (0 px) are not fitted; how often the real detector still finds them is reported.
Validation: leave-one-layout-out log loss and calibration by group x altitude, against the single curve
currently used (a50 339.6 px, slope 1.0, p_max 0.85).

    ~/isaacsim/python.sh isaac/fit_detection_model.py \
        --layout open=isaac/out/frames_open_real --layout booth=isaac/out/frames_booth_real \
        --layout cubicle=isaac/out/frames_cubicle_real
Prints the report and the coefficients as JSON (paste them back) and writes isaac/results/DETECTION_MODEL.md/.json.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)

import export_scene as ES  # noqa: E402
from lbyf.config import SimConfig  # noqa: E402
from lbyf.visibility import GROUP_OF, VisParams, footprint, visible_pixels  # noqa: E402
from lbyf.world import make_world  # noqa: E402

GROUPS = ("tall", "medium", "flat", "small")
CURRENT = {"a50": 339.6, "slope": 1.0, "pmax": 0.85}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, l2=1.0, iters=50):
    """Newton / IRLS logistic regression with a small ridge penalty (not on the intercept)."""
    w = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1])
    R[0, 0] = 0.0
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) + R @ w
        H = X.T @ (X * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def rows_for_layout(name, frames_dir, prm):
    cfg = SimConfig(layout=name)
    spec = ES.build_spec(cfg)
    world = make_world(cfg.seed, **cfg.world_kwargs())
    with open(os.path.join(frames_dir, "detection_rows.json")) as fh:
        det = json.load(fh)
    vp_xy = {v["node"]: (v["x"], v["y"]) for v in spec["viewpoints"]}
    obj = {o["oid"]: o for o in spec["objects"]}
    out = []
    for r in det["rows"]:
        if r["node"] not in vp_xy or r["oid"] not in obj:
            raise SystemExit(f"{name}: detection rows do not match the regenerated scene (default seed/time?)")
        ob = obj[r["oid"]]
        (x, y), alt = vp_xy[r["node"]], r["alt"]
        w, h, top = footprint(ob["shape"], ob["dims"])
        px = visible_pixels(world, (x, y, alt), ob["x"], ob["y"], ob["z_base"], w, h, top, prm)
        cz = ob["z_base"] + h / 2
        down = math.degrees(math.atan2(alt - cz, max(math.hypot(ob["x"] - x, ob["y"] - y), 1e-3)))
        out.append({"layout": name, "alt": alt, "cls": r["cls"], "group": GROUP_OF[r["cls"]], "sim_px": px,
                    "down": down, "det": float(r["det"]), "render_px": r["px"]})
    return out


def design(rows, angle=True):
    lp = np.log(np.maximum([r["sim_px"] for r in rows], 1.0))
    d = np.array([max(r["down"], 0.0) / 30.0 for r in rows])
    cols = [np.ones(len(rows)), lp] + ([d] if angle else [])
    return np.stack(cols, 1)


def fit_groups(rows, angle=True):
    coef = {}
    for g in GROUPS:
        sub = [r for r in rows if r["group"] == g and r["sim_px"] > 0]
        if len(sub) < 30 or len({r["det"] for r in sub}) < 2:
            coef[g] = None
            continue
        coef[g] = fit_logreg(design(sub, angle), np.array([r["det"] for r in sub])).tolist()
    return coef


def predict(rows, coef, angle=True):
    p = np.zeros(len(rows))
    for i, r in enumerate(rows):
        if r["sim_px"] <= 0:
            continue
        c = coef.get(r["group"])
        if c is None:
            p[i] = current_curve(r["sim_px"])
            continue
        x = design([r], angle)[0]
        p[i] = sigmoid(x @ np.array(c))
    return p


def current_curve(px):
    return CURRENT["pmax"] / (1 + math.exp(-(math.log(max(px, 1e-9)) - math.log(CURRENT["a50"])) / CURRENT["slope"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", action="append", required=True, help="name=frames_dir (repeat)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "DETECTION_MODEL.md"))
    args = ap.parse_args()
    prm = VisParams(use_top=True, use_vfov=True)
    rows = []
    for spec_arg in args.layout:
        name, d = spec_arg.split("=", 1)
        rows += rows_for_layout(name, d, prm)
    names = sorted({r["layout"] for r in rows})
    vis = [r for r in rows if r["sim_px"] > 0]
    y = np.array([r["det"] for r in vis])

    L = ["# Fitted detection model: simulator pixels and viewing angle -> real detector (YOLO-World)\n",
         f"Layouts {', '.join(names)}; {len(rows)} viewpoint-object pairs, {len(vis)} the simulator calls visible. "
         f"Real detections on pairs the simulator calls invisible: "
         f"{sum(r['det'] for r in rows if r['sim_px'] <= 0):.0f} of {sum(1 for r in rows if r['sim_px'] <= 0)}.\n",
         "**Leave-one-layout-out log loss** (lower is better; fitted on the other layouts):\n",
         "| Held-out layout | Current single curve | Per group | Per group + viewing angle |", "|---|---|---|---|"]
    for held in names:
        tr = [r for r in rows if r["layout"] != held]
        te = [r for r in vis if r["layout"] == held]
        yt = np.array([r["det"] for r in te])
        base = np.array([current_curve(r["sim_px"]) for r in te])
        p1 = predict(te, fit_groups(tr, angle=False), angle=False)
        p2 = predict(te, fit_groups(tr, angle=True), angle=True)
        L.append(f"| {held} | {logloss(base, yt):.4f} | {logloss(p1, yt):.4f} | {logloss(p2, yt):.4f} |")

    coef = fit_groups(rows, angle=True)
    p_all = predict(vis, coef, angle=True)
    base_all = np.array([current_curve(r["sim_px"]) for r in vis])
    L += ["\n**Coefficients** (all layouts): p = sigmoid(b0 + b1·ln(sim px) + b2·down/30°)\n",
          "| Group | Classes | b0 | b1 | b2 (per 30° looking down) | Pairs |", "|---|---|---|---|---|---|"]
    for g in GROUPS:
        c = coef[g]
        n = sum(1 for r in vis if r["group"] == g)
        members = ", ".join(sorted(k for k, v in GROUP_OF.items() if v == g))
        L.append(f"| {g} | {members} | " + (" | ".join(f"{v:.3f}" for v in c) if c else "— | — | —") + f" | {n} |")
    L += ["\n**Calibration by group and altitude** (mean real detection / fitted / current curve, pairs the simulator calls visible):\n",
          "| Group | " + " | ".join(f"{a:g} m" for a in sorted({r['alt'] for r in vis})) + " |",
          "|---|" + "---|" * len({r['alt'] for r in vis})]
    for g in GROUPS:
        cells = []
        for a in sorted({r["alt"] for r in vis}):
            idx = [i for i, r in enumerate(vis) if r["group"] == g and r["alt"] == a]
            cells.append(f"{y[idx].mean():.2f} / {p_all[idx].mean():.2f} / {base_all[idx].mean():.2f}" if idx else "—")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    text = "\n".join(L) + "\n"
    payload = {"model": "logit(b0 + b1 ln px + b2 down/30)", "coef": coef, "layouts": names,
               "n_pairs": len(rows)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(payload, fh, indent=1)
    print(text)
    print("PASTE THIS BACK:")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
