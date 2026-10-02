"""Stage 3b: run a real open-vocabulary detector on rendered frames and measure when it finds each object.

Inputs come from capture.py run with every RGB frame and segmentation saved:
    ~/isaacsim/python.sh isaac/capture.py --spec S --out F --headless --assets real --rgb-every 1 --save-seg
    ~/isaacsim/python.sh isaac/detect_frames.py --spec S --frames F                # detect + report
    ~/isaacsim/python.sh isaac/detect_frames.py --spec S --frames F --from-raw     # re-report, no detector run

Each class has several text prompts (synonyms); a detection's class is the class of its prompt. Ground truth per
(frame, object): visible pixels and tight box from instance segmentation. A detection matches an object when the
class agrees and the box overlaps it (IoU >= 0.3, or >= 50 % of the object's pixels inside the box); greedy by
confidence. Per viewpoint (any of the 4 yaws, as in the visibility model) we get (visible pixels, detected?).

The operating confidence is the lowest of CONF_SWEEP giving <= --max-fp false positives per frame (unless --conf is
given). The report has a confidence sweep, detection rate vs visible pixels, a fitted
p(detect | pixels) = p_max * logistic((ln px - ln a50) / slope) (the simulator's form), per-class and
class x altitude tables, and a confusion table (what label, if any, clearly visible objects got).
Writes <frames>/detections_raw.json, DETECTION.md, detection_rows.json, detection_fit.json, det_viz_*.png.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

# class -> prompts; the toolbox class is a briefcase model (the asset library has no toolbox)
PROMPTS = {"chair": ["office chair", "chair"], "stool": ["stool", "bar stool"],
           "cart": ["cart", "trolley", "utility cart"], "bag": ["bag", "handbag", "medical bag"],
           "laptop": ["laptop", "closed laptop", "notebook computer"],
           "monitor": ["monitor", "computer monitor", "screen", "tv"], "box": ["cardboard box", "box"],
           "bin": ["trash can", "waste bin"], "plant": ["potted plant", "plant"],
           "toolbox": ["briefcase", "suitcase"], "mug": ["mug", "cup", "coffee mug"]}
PROMPT_LIST = [(c, p) for c, ps in PROMPTS.items() for p in ps]
OBJ_RE = re.compile(r"/World/Objects/obj_(\d+)")
PX_BINS = [1, 50, 200, 800, 3200, 12800, 10 ** 9]
CONF_SWEEP = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35)
CLEAR_PX = 1600  # "clearly visible" for the confusion table


def gt_objects(npz_path):
    """{oid: (pixels, (x0, y0, x1, y1), mask)} from a saved segmentation."""
    z = np.load(npz_path, allow_pickle=False)
    ids = z["ids"]
    id_to_path = json.loads(str(z["id_to_path"]))
    by_oid = defaultdict(list)
    for k, v in id_to_path.items():
        m = OBJ_RE.search(str(v))
        if m:
            by_oid[int(m.group(1))].append(int(k))
    out = {}
    for oid, keys in by_oid.items():
        mask = np.isin(ids, keys)
        n = int(mask.sum())
        if n:
            ys, xs = np.nonzero(mask)
            out[oid] = (n, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), mask)
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlaps(dets, gts):
    """{oid: [(det index, overlap score)]} for every detection box that overlaps an object enough to match it."""
    ov = defaultdict(list)
    for j, (_, _, box) in enumerate(dets):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        for oid, (n, gbox, mask) in gts.items():
            inside = mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)].sum() / n
            score = iou(box, gbox)
            if score >= 0.3 or inside >= 0.5:
                ov[oid].append((j, float(max(score, inside))))
    return dict(ov)


def match_from(dets, ov, cls_of, conf=0.0):
    """Greedy one-to-one matching by confidence among detections >= conf; returns {oid: confidence}."""
    by_det = defaultdict(list)
    for oid, lst in ov.items():
        for j, score in lst:
            by_det[j].append((score, oid))
    matched = {}
    for j in sorted(range(len(dets)), key=lambda k: -dets[k][1]):
        cls, c = dets[j][0], dets[j][1]
        if c < conf:
            break
        cand = [(sc, oid) for sc, oid in by_det.get(j, []) if oid not in matched and cls_of.get(oid) == cls]
        if cand:
            matched[max(cand)[1]] = c
    return matched


def match(dets, gts, cls_of):
    """dets: [(cls, conf, box)]; returns {oid: confidence of its best matching detection}."""
    return match_from(dets, overlaps(dets, gts), cls_of)


def fit_logistic(px, hit):
    """Grid MLE of p_max * logistic((ln px - ln a50) / slope)."""
    px, hit = np.maximum(np.asarray(px, float), 1.0), np.asarray(hit, float)
    lx = np.log(px)
    best = None
    for pmax in (0.80, 0.85, 0.90, 0.95, 0.99):
        for a50 in np.geomspace(10, 20000, 70):
            for slope in (0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4):
                p = np.clip(pmax / (1 + np.exp(-(lx - math.log(a50)) / slope)), 1e-6, 1 - 1e-6)
                ll = float(np.sum(hit * np.log(p) + (1 - hit) * np.log(1 - p)))
                if best is None or ll > best[0]:
                    best = (ll, pmax, float(a50), slope)
    return {"p_max": best[1], "a50_px": round(best[2], 1), "slope": best[3], "loglik": round(best[0], 1)}


def run_detector(args, frames, cls_of):
    from ultralytics import YOLOWorld
    model = YOLOWorld(args.model)
    model.set_classes([p for _, p in PROMPT_LIST])
    raw, viz_left = [], args.viz
    for i, (f, rgb, seg) in enumerate(frames):
        res = model.predict(rgb, conf=args.min_conf, imgsz=args.imgsz, verbose=False)[0]
        dets = [(PROMPT_LIST[int(k)][0], float(sc), [round(float(v), 1) for v in b], PROMPT_LIST[int(k)][1])
                for b, sc, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                    res.boxes.cls.cpu().numpy())]
        gts = gt_objects(seg)
        ov = overlaps([d[:3] for d in dets], gts)
        raw.append({"frame": f["frame"], "node": f["node"], "alt": f["alt"], "yaw": f["yaw"],
                    "dets": [[d[0], round(d[1], 4), d[2], d[3]] for d in dets],
                    "gt": {str(o): [g[0], list(g[1])] for o, g in gts.items()},
                    "ov": {str(o): [[j, round(sc, 3)] for j, sc in lst] for o, lst in ov.items()}})
        if viz_left > 0 and len(gts) >= 3 and i % 7 == 0:
            im = Image.open(rgb).convert("RGB")
            d = ImageDraw.Draw(im)
            for oid, (n, gb, _) in gts.items():
                d.rectangle(gb, outline=(255, 255, 255), width=1)
                d.text((gb[0] + 2, gb[3] - 12), f"#{oid} {cls_of[oid]}", fill=(255, 255, 255))
            for cls, conf, b, prompt in dets:
                if conf >= 0.1:
                    d.rectangle(b, outline=(0, 255, 0), width=3)
                    d.text((b[0] + 3, b[1] + 3), f"{prompt} {conf:.2f}", fill=(0, 255, 0))
            im.save(os.path.join(args.frames, f"det_viz_{f['frame']:05d}.png"))
            viz_left -= 1
        if i % 50 == 0 or i == len(frames) - 1:
            print(f"[detect] frame {i + 1}/{len(frames)}: {len(dets)} detections, {len(gts)} objects in view")
    return raw


def per_viewpoint(raw, cls_of, conf):
    """Rows (node, alt, oid, cls, max px, detected, proposed) at a confidence threshold, plus per-frame means of
    false positives (detections not matching their class) and background false positives (boxes on no object).
    detected = one-to-one match with the right class; proposed = any detection >= conf overlaps the object,
    whatever its label (what a detector + re-identification pipeline needs)."""
    per_vp, fps, bg = {}, [], []
    for fr in raw:
        dets = [tuple(d[:3]) for d in fr["dets"]]
        ov = {int(o): [tuple(x) for x in lst] for o, lst in fr["ov"].items()}
        matched = match_from(dets, ov, cls_of, conf)
        on_obj = {j for lst in ov.values() for j, _ in lst}
        n_conf = sum(1 for d in dets if d[1] >= conf)
        fps.append(n_conf - len(matched))
        bg.append(sum(1 for j, d in enumerate(dets) if d[1] >= conf and j not in on_obj))
        for oid in cls_of:
            key = (fr["node"], round(fr["alt"], 3), oid)
            px = fr["gt"].get(str(oid), [0])[0]
            prop = any(dets[j][1] >= conf for j, _ in ov.get(oid, []))
            cur = per_vp.setdefault(key, [0, False, False])
            cur[0], cur[1], cur[2] = max(cur[0], px), cur[1] or oid in matched, cur[2] or prop
    rows = [{"node": k[0], "alt": k[1], "oid": k[2], "cls": cls_of[k[2]], "px": v[0], "det": v[1], "prop": v[2]}
            for k, v in per_vp.items()]
    return rows, float(np.mean(fps)), float(np.mean(bg))


def confusion(raw, cls_of):
    """For clearly visible objects (>= CLEAR_PX in a frame): label of the most confident overlapping detection."""
    out = defaultdict(Counter)
    for fr in raw:
        for o, (px, _) in fr["gt"].items():
            if px < CLEAR_PX:
                continue
            cands = [fr["dets"][j] for j, sc in fr["ov"].get(o, [])]
            label = max(cands, key=lambda d: d[1])[3] if cands else "(nothing)"
            out[cls_of[int(o)]][label] += 1
    return out


def report(spec, index, raw, cls_of, args):
    sweep = {c: per_viewpoint(raw, cls_of, c) for c in CONF_SWEEP}
    conf = args.conf if args.conf is not None else next(
        (c for c in CONF_SWEEP if sweep[c][2] <= args.max_fp), CONF_SWEEP[-1])
    rows, fp, bg = sweep[conf] if conf in sweep else per_viewpoint(raw, cls_of, conf)
    vis = [r for r in rows if r["px"] > 0]
    alts = sorted({r["alt"] for r in rows})
    names = list(PROMPTS)
    L = ["# Stage 3b: real detector on Isaac Sim renders\n",
         f"Layout `{spec['meta']['layout']}`, {len(raw)} frames, assets `{index.get('assets', 'proxy')}`, detector "
         f"`{args.model}` with {len(PROMPT_LIST)} prompts for {len(PROMPTS)} classes. Per viewpoint = best over the yaws.\n",
         "Two scores: **detected** = a detection with the right class label matches the object; **proposed** = any "
         "detection overlaps it, whatever its label (enough when re-identification decides identity). False positives "
         "are split into *background* (a box on no object) and *all* (also counting real objects given a wrong label).\n",
         f"**Operating confidence {conf}** ({'given' if args.conf is not None else f'lowest with ≤ {args.max_fp} background false positives per frame'}): "
         f"{bg:.2f} background / {fp:.2f} all false positives per frame.\n",
         "**Confidence sweep:**\n",
         "| Confidence ≥ | FP / frame (background / all) | Detected, ≥ 800 px | Proposed, ≥ 800 px | " +
         " | ".join(f"Detected / proposed per viewpoint @{a:g} m" for a in alts) + " |",
         "|---|---|---|---|" + "---|" * len(alts)]
    for c in CONF_SWEEP:
        r_c, fp_c, bg_c = sweep[c]
        big = [r for r in r_c if r["px"] >= 800]
        cells = []
        for a in alts:
            sub = [r for r in r_c if r["alt"] == a]
            n = len({r["node"] for r in sub})
            cells.append(f"{sum(r['det'] for r in sub) / n:.1f} / {sum(r['prop'] for r in sub) / n:.1f}")
        L.append(f"| {c} | {bg_c:.2f} / {fp_c:.2f} | {np.mean([r['det'] for r in big]):.1%} | "
                 f"{np.mean([r['prop'] for r in big]):.1%} | " + " | ".join(cells) + " |")
    L += ["\n**Detection rate vs visible pixels** (pairs with ≥ 1 visible pixel):\n",
          "| Visible pixels | Pairs | Detected | Proposed |", "|---|---|---|---|"]
    for lo, hi in zip(PX_BINS[:-1], PX_BINS[1:]):
        sub = [r for r in vis if lo <= r["px"] < hi]
        if sub:
            L.append(f"| {lo}–{hi if hi < 10 ** 9 else '∞'} | {len(sub)} | {np.mean([r['det'] for r in sub]):.1%} | "
                     f"{np.mean([r['prop'] for r in sub]):.1%} |")
    fit = fit_logistic([r["px"] for r in vis], [r["det"] for r in vis]) if len(vis) >= 30 else None
    fit_p = fit_logistic([r["px"] for r in vis], [r["prop"] for r in vis]) if len(vis) >= 30 else None
    if fit:
        L.append(f"\n**Fitted curves (all classes):** detected: p_max {fit['p_max']}, a50 **{fit['a50_px']} px**, slope "
                 f"{fit['slope']}; proposed: p_max {fit_p['p_max']}, a50 **{fit_p['a50_px']} px**, slope {fit_p['slope']}. "
                 f"The simulator's render-pixel calibration used a50 = 224.5 px, slope 0.35, p_max 0.95.\n")
    L += ["**Per class:**\n",
          "| Class | Pairs (≥ 1 px) | Detected | Detected when ≥ 800 px | Proposed when ≥ 800 px | Fitted a50, detected (px) |",
          "|---|---|---|---|---|---|"]
    per_class_fit = {}
    for c in names:
        sub = [r for r in vis if r["cls"] == c]
        if not sub:
            continue
        big = [r for r in sub if r["px"] >= 800]
        f_c = fit_logistic([r["px"] for r in sub], [r["det"] for r in sub]) if len(sub) >= 30 else None
        per_class_fit[c] = f_c
        big_rate = f"{np.mean([r['det'] for r in big]):.1%} ({len(big)})" if big else "—"
        big_prop = f"{np.mean([r['prop'] for r in big]):.1%}" if big else "—"
        L.append(f"| {c} | {len(sub)} | {np.mean([r['det'] for r in sub]):.1%} | {big_rate} | {big_prop} | "
                 f"{f_c['a50_px'] if f_c else '—'} |")
    L += ["\n**What clearly visible objects (≥ %d px in a frame) were labelled as** (most confident overlapping "
          "detection at confidence ≥ %g, any class; top 4 labels):\n" % (CLEAR_PX, args.min_conf),
          "| Class | Frames | Labels |", "|---|---|---|"]
    for c, cnt in sorted(confusion(raw, cls_of).items()):
        n = sum(cnt.values())
        L.append(f"| {c} | {n} | " + ", ".join(f"{lab} {k / n:.0%}" for lab, k in cnt.most_common(4)) + " |")
    L += ["\n**Visible (≥ 200 px) / detected rate by class and altitude:**\n",
          "| Class | " + " | ".join(f"{a:g} m" for a in alts) + " |", "|---|" + "---|" * len(alts)]
    for c in names:
        cells = []
        for a in alts:
            sub = [r for r in rows if r["cls"] == c and r["alt"] == a]
            cells.append(f"{np.mean([r['px'] >= 200 for r in sub]):.1%} / {np.mean([r['det'] for r in sub]):.1%}"
                         if sub else "—")
        if any(cl != "—" for cl in cells):
            L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["\n**Objects per viewpoint, by altitude:**\n", "| Altitude | Visible (≥ 200 px) | Detected | Proposed |",
          "|---|---|---|---|"]
    for a in alts:
        sub = [r for r in rows if r["alt"] == a]
        n_vp = len({r["node"] for r in sub})
        L.append(f"| {a:g} m | {sum(r['px'] >= 200 for r in sub) / n_vp:.1f} | {sum(r['det'] for r in sub) / n_vp:.1f} | "
                 f"{sum(r['prop'] for r in sub) / n_vp:.1f} |")
    return "\n".join(L) + "\n", rows, conf, {"detected": fit, "proposed": fit_p}, per_class_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--model", default="yolov8x-worldv2.pt")
    ap.add_argument("--conf", type=float, default=None, help="operating confidence (default: chosen by --max-fp)")
    ap.add_argument("--max-fp", type=float, default=0.5, help="background false positives per frame allowed at the operating point")
    ap.add_argument("--min-conf", type=float, default=0.05, help="lowest confidence kept from the detector")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--viz", type=int, default=8, help="annotated example images to write")
    ap.add_argument("--from-raw", action="store_true", help="reuse detections_raw.json instead of running the detector")
    args = ap.parse_args()

    with open(args.spec) as fh:
        spec = json.load(fh)
    with open(os.path.join(args.frames, "index.json")) as fh:
        index = json.load(fh)
    cls_of = {o["oid"]: o["cls"] for o in spec["objects"]}
    raw_path = os.path.join(args.frames, "detections_raw.json")
    if args.from_raw:
        with open(raw_path) as fh:
            raw = json.load(fh)["frames"]
    else:
        frames = []
        for f in index["frames"]:
            rgb = os.path.join(args.frames, f"rgb_{f['frame']:05d}.png")
            seg = os.path.join(args.frames, f"seg_{f['frame']:05d}.npz")
            if os.path.exists(rgb) and os.path.exists(seg):
                frames.append((f, rgb, seg))
        if not frames:
            raise SystemExit("no frames with both rgb_*.png and seg_*.npz - run capture.py with --rgb-every 1 --save-seg")
        print(f"[detect] {len(frames)} frames, assets={index.get('assets', 'proxy')}, model {args.model}")
        raw = run_detector(args, frames, cls_of)
        with open(raw_path, "w") as fh:
            json.dump({"model": args.model, "prompts": PROMPT_LIST, "frames": raw}, fh)

    text, rows, conf, fit, per_class_fit = report(spec, index, raw, cls_of, args)
    with open(os.path.join(args.frames, "DETECTION.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(args.frames, "detection_rows.json"), "w") as fh:
        json.dump({"conf": conf, "model": args.model, "assets": index.get("assets", "proxy"), "rows": rows}, fh)
    with open(os.path.join(args.frames, "detection_fit.json"), "w") as fh:
        json.dump({"all": fit, "per_class": per_class_fit, "conf": conf, "model": args.model}, fh, indent=1)
    print(text)
    print(f"[detect] wrote {os.path.join(args.frames, 'DETECTION.md')}; examples: "
          f"{sorted(glob.glob(os.path.join(args.frames, 'det_viz_*.png')))[:4]}")


if __name__ == "__main__":
    main()
