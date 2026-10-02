"""Stage 3a analysis: does the Stage 1-2 visibility model agree with what Isaac Sim actually renders?

Plain Python + numpy (no Isaac Sim). For each (viewpoint, altitude, object):
    model says visible   <=> model detection probability >= 0.5   (line of sight + range, from export_scene.py)
    render says visible  <=> at least --min-pixels pixels of the object in any yaw at that viewpoint

Pre-registered rule: if overall agreement < 80 %, the Stage 1-2 visibility model must be revised and Stage 2
re-run before its conclusions are trusted.

    python isaac/analyze_visibility.py --spec isaac/out/scene_spec.json --frames isaac/out/frames
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

RULE = 0.80


def load(spec_path, frames_dir):
    with open(spec_path) as fh:
        spec = json.load(fh)
    with open(os.path.join(frames_dir, "index.json")) as fh:
        index = json.load(fh)
    return spec, index


def compare(spec, index, min_pixels=200, model_threshold=0.5):
    """Rows of (node, alt, oid, cls, model_visible, render_visible, max_pixels) for every rendered viewpoint."""
    seen = defaultdict(int)
    rendered = set()
    for f in index["frames"]:
        key = (f["node"], round(f["alt"], 3))
        rendered.add(key)
        for oid, px in f["pixels"].items():
            seen[key + (int(oid),)] = max(seen[key + (int(oid),)], int(px))
    cls_of = {o["oid"]: o["cls"] for o in spec["objects"]}
    rows = []
    for vp in spec["viewpoints"]:
        key = (vp["node"], round(vp["alt"], 3))
        if key not in rendered:
            continue
        for oid_s, p in vp["model_p"].items():
            oid = int(oid_s)
            px = seen.get(key + (oid,), 0)
            rows.append((vp["node"], vp["alt"], oid, cls_of[oid], p >= model_threshold, px >= min_pixels, px))
    return rows


def kappa(m, r):
    m, r = np.asarray(m, bool), np.asarray(r, bool)
    po = np.mean(m == r)
    pe = m.mean() * r.mean() + (1 - m.mean()) * (1 - r.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def summarise(rows):
    m = np.array([r[4] for r in rows])
    r = np.array([r[5] for r in rows])
    out = {"n": len(rows), "agreement": float(np.mean(m == r)), "kappa": kappa(m, r),
           "model_visible_rate": float(m.mean()), "render_visible_rate": float(r.mean()),
           "render_given_model": float(r[m].mean()) if m.any() else np.nan,
           "render_given_not_model": float(r[~m].mean()) if (~m).any() else np.nan}
    return out


def report(spec, rows, min_pixels):
    L = [f"# Stage 3a: visibility model vs Isaac Sim render\n",
         f"Layout `{spec['meta']['layout']}`, seed {spec['meta']['seed']}, {len(spec['objects'])} objects. "
         f"Render-visible = at least {min_pixels} pixels in any of the yaws at a viewpoint. "
         f"Model-visible = detection probability ≥ 0.5.\n"]
    s = summarise(rows)
    verdict = "PASS" if s["agreement"] >= RULE else "FAIL"
    L.append(f"**Pre-registered rule (agreement ≥ {RULE:.0%}): {verdict}.** Agreement {s['agreement']:.1%}, "
             f"Cohen's κ {s['kappa']:.2f}, over {s['n']} viewpoint–object pairs.\n")
    L.append("| Slice | Pairs | Agreement | κ | Model says visible | Render says visible | Rendered when model says visible | Rendered when model says hidden |")
    L.append("|---|---|---|---|---|---|---|---|")

    def line(name, sub):
        if not sub:
            return
        t = summarise(sub)
        L.append(f"| {name} | {t['n']} | {t['agreement']:.1%} | {t['kappa']:.2f} | {t['model_visible_rate']:.1%} | "
                 f"{t['render_visible_rate']:.1%} | {t['render_given_model']:.1%} | {t['render_given_not_model']:.1%} |")

    line("all", rows)
    for alt in sorted({r[1] for r in rows}):
        line(f"altitude {alt:g} m", [r for r in rows if r[1] == alt])
    for c in sorted({r[3] for r in rows}):
        line(f"class {c}", [r for r in rows if r[3] == c])

    L.append("\n**Objects visible per viewpoint, by altitude** (the quantity behind the Stage 2 altitude result):\n")
    L.append("| Altitude | Model | Render |")
    L.append("|---|---|---|")
    for alt in sorted({r[1] for r in rows}):
        sub = [r for r in rows if r[1] == alt]
        nodes = {r[0] for r in sub}
        L.append(f"| {alt:g} m | {sum(r[4] for r in sub) / len(nodes):.1f} | {sum(r[5] for r in sub) / len(nodes):.1f} |")
    return "\n".join(L) + "\n", verdict, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--min-pixels", type=int, default=200)
    ap.add_argument("--out", default=None, help="default: <frames>/VISIBILITY.md")
    args = ap.parse_args()
    spec, index = load(args.spec, args.frames)
    rows = compare(spec, index, args.min_pixels)
    text, verdict, s = report(spec, rows, args.min_pixels)
    out = args.out or os.path.join(args.frames, "VISIBILITY.md")
    with open(out, "w") as fh:
        fh.write(text)
    print(text)
    print(f"verdict: {verdict} | wrote {out}")


if __name__ == "__main__":
    main()
