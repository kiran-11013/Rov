"""Stage 3b: run a real open-vocabulary detector on rendered frames and measure when it finds each object.

Inputs come from capture.py run with every RGB frame and segmentation saved:
    ~/isaacsim/python.sh isaac/capture.py --spec S --out F --headless --assets real --rgb-every 1 --save-seg
    ~/isaacsim/python.sh isaac/detect_frames.py --spec S --frames F

Ground truth per (frame, object): visible pixels and tight box from instance segmentation. A detection matches
an object when the class agrees and the box overlaps it (IoU >= 0.3, or >= 50 % of the object's pixels inside
the box); greedy by confidence. Per viewpoint (any of the 4 yaws, as in the visibility model) we then have
(visible pixels, best confidence), from which the report gives:
    * detection rate vs visible pixels, overall and per class
    * a fitted p(detect | pixels) = p_max * logistic((ln px - ln a50) / slope), the same form the simulator uses
    * visible vs detected rate by class and altitude, and false positives per frame
Writes <frames>/DETECTION.md, <frames>/detection_rows.json and a few annotated images (<frames>/det_viz_*.png).
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

PROMPTS = {"chair": "office chair", "stool": "stool", "cart": "cart", "bag": "bag", "laptop": "laptop",
           "monitor": "computer monitor", "box": "cardboard box", "bin": "trash can", "plant": "potted plant",
           "toolbox": "briefcase", "mug": "mug"}  # toolbox class is a briefcase model (no toolbox in the library)
OBJ_RE = re.compile(r"/World/Objects/obj_(\d+)")
PX_BINS = [1, 50, 200, 800, 3200, 12800, 10 ** 9]


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


def match(dets, gts, cls_of):
    """dets: [(cls, conf, box)]; returns {oid: confidence of its best matching detection}."""
    matched = {}
    for cls, conf, box in sorted(dets, key=lambda d: -d[1]):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cand = []
        for oid, (n, gbox, mask) in gts.items():
            if oid in matched or cls_of.get(oid) != cls:
                continue
            inside = mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)].sum() / n
            overlap = iou(box, gbox)
            if overlap >= 0.3 or inside >= 0.5:
                cand.append((max(overlap, inside), oid))
        if cand:
            matched[max(cand)[1]] = conf
    return matched


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--model", default="yolov8x-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence counted as 'detected'")
    ap.add_argument("--min-conf", type=float, default=0.05, help="lowest confidence kept from the detector")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--viz", type=int, default=6, help="annotated example images to write")
    args = ap.parse_args()

    with open(args.spec) as fh:
        spec = json.load(fh)
    with open(os.path.join(args.frames, "index.json")) as fh:
        index = json.load(fh)
    cls_of = {o["oid"]: o["cls"] for o in spec["objects"]}
    frames = []
    for f in index["frames"]:
        rgb = os.path.join(args.frames, f"rgb_{f['frame']:05d}.png")
        seg = os.path.join(args.frames, f"seg_{f['frame']:05d}.npz")
        if os.path.exists(rgb) and os.path.exists(seg):
            frames.append((f, rgb, seg))
    if not frames:
        raise SystemExit("no frames with both rgb_*.png and seg_*.npz - run capture.py with --rgb-every 1 --save-seg")
    print(f"[detect] {len(frames)} frames, assets={index.get('assets', 'proxy')}, model {args.model}")

    from ultralytics import YOLOWorld
    model = YOLOWorld(args.model)
    names = list(PROMPTS)
    model.set_classes([PROMPTS[c] for c in names])

    per_vp = {}  # (node, alt, oid) -> [max px, max conf]
    fp_per_frame, viz_left = [], args.viz
    for i, (f, rgb, seg) in enumerate(frames):
        res = model.predict(rgb, conf=args.min_conf, imgsz=args.imgsz, verbose=False)[0]
        dets = [(names[int(c)], float(s), b.tolist()) for b, s, c in
                zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy())]
        gts = gt_objects(seg)
        matched = match(dets, gts, cls_of)
        fp_per_frame.append(sum(1 for d in dets if d[1] >= args.conf) - sum(1 for c in matched.values() if c >= args.conf))
        for oid in cls_of:
            key = (f["node"], round(f["alt"], 3), oid)
            px = gts[oid][0] if oid in gts else 0
            cur = per_vp.setdefault(key, [0, 0.0])
            cur[0], cur[1] = max(cur[0], px), max(cur[1], matched.get(oid, 0.0))
        if viz_left > 0 and len(gts) >= 3 and i % 7 == 0:
            im = Image.open(rgb).convert("RGB")
            d = ImageDraw.Draw(im)
            for oid, (n, gb, _) in gts.items():
                d.rectangle(gb, outline=(255, 255, 255), width=1)
            for cls, conf, b in dets:
                if conf >= args.conf:
                    d.rectangle(b, outline=(0, 255, 0), width=3)
                    d.text((b[0] + 3, b[1] + 3), f"{cls} {conf:.2f}", fill=(0, 255, 0))
            im.save(os.path.join(args.frames, f"det_viz_{f['frame']:05d}.png"))
            viz_left -= 1
        if i % 50 == 0 or i == len(frames) - 1:
            print(f"[detect] frame {i + 1}/{len(frames)}: {len(dets)} detections, {len(gts)} objects in view")

    rows = [{"node": k[0], "alt": k[1], "oid": k[2], "cls": cls_of[k[2]], "px": v[0], "conf": round(v[1], 3)}
            for k, v in per_vp.items()]
    with open(os.path.join(args.frames, "detection_rows.json"), "w") as fh:
        json.dump({"conf": args.conf, "model": args.model, "assets": index.get("assets", "proxy"), "rows": rows}, fh)

    vis = [r for r in rows if r["px"] > 0]
    hit = lambda r: r["conf"] >= args.conf  # noqa: E731
    L = [f"# Stage 3b: real detector on Isaac Sim renders\n",
         f"Layout `{spec['meta']['layout']}`, {len(frames)} frames, assets `{index.get('assets', 'proxy')}`, "
         f"detector `{args.model}` (open-vocabulary prompts), detected = matched with confidence ≥ {args.conf}. "
         f"Per viewpoint = best over the yaws. False positives: {np.mean(fp_per_frame):.2f} per frame.\n",
         "**Detection rate vs visible pixels** (viewpoint–object pairs with at least 1 visible pixel):\n",
         "| Visible pixels | Pairs | Detected |", "|---|---|---|"]
    for lo, hi in zip(PX_BINS[:-1], PX_BINS[1:]):
        sub = [r for r in vis if lo <= r["px"] < hi]
        if sub:
            L.append(f"| {lo}–{hi if hi < 10 ** 9 else '∞'} | {len(sub)} | {np.mean([hit(r) for r in sub]):.1%} |")
    fit = fit_logistic([r["px"] for r in vis], [hit(r) for r in vis]) if len(vis) >= 30 else None
    if fit:
        L.append(f"\n**Fitted detection curve (all classes):** p_max {fit['p_max']}, a50 **{fit['a50_px']} px**, slope "
                 f"{fit['slope']}. The simulator's render-pixel calibration used a50 = 224.5 px, slope 0.35, p_max 0.95.\n")
    L += ["**Per class** (pairs with ≥ 1 visible pixel; fitted a50 where there are enough pairs):\n",
          "| Class | Pairs | Detected | Detected when ≥ 200 px | Fitted a50 (px) |", "|---|---|---|---|---|"]
    per_class_fit = {}
    for c in names:
        sub = [r for r in vis if r["cls"] == c]
        if not sub:
            continue
        big = [r for r in sub if r["px"] >= 200]
        f_c = fit_logistic([r["px"] for r in sub], [hit(r) for r in sub]) if len(sub) >= 30 else None
        per_class_fit[c] = f_c
        big_rate = f"{np.mean([hit(r) for r in big]):.1%}" if big else "—"
        L.append(f"| {c} | {len(sub)} | {np.mean([hit(r) for r in sub]):.1%} | {big_rate} | "
                 f"{f_c['a50_px'] if f_c else '—'} |")
    alts = sorted({r["alt"] for r in rows})
    L += ["\n**Visible (≥ 200 px) / detected rate by class and altitude** (all pairs):\n",
          "| Class | " + " | ".join(f"{a:g} m" for a in alts) + " |", "|---|" + "---|" * len(alts)]
    for c in names:
        cells = []
        for a in alts:
            sub = [r for r in rows if r["cls"] == c and r["alt"] == a]
            cells.append(f"{np.mean([r['px'] >= 200 for r in sub]):.1%} / {np.mean([hit(r) for r in sub]):.1%}"
                         if sub else "—")
        if any(cl != "—" for cl in cells):
            L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["\n**Objects detected per viewpoint, by altitude:**\n", "| Altitude | Visible (≥ 200 px) | Detected |",
          "|---|---|---|"]
    for a in alts:
        sub = [r for r in rows if r["alt"] == a]
        n_vp = len({r["node"] for r in sub})
        L.append(f"| {a:g} m | {sum(r['px'] >= 200 for r in sub) / n_vp:.1f} | {sum(hit(r) for r in sub) / n_vp:.1f} |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(args.frames, "DETECTION.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(args.frames, "detection_fit.json"), "w") as fh:
        json.dump({"all": fit, "per_class": per_class_fit, "conf": args.conf, "model": args.model}, fh, indent=1)
    print(text)
    print(f"[detect] wrote {os.path.join(args.frames, 'DETECTION.md')}; examples: "
          f"{sorted(glob.glob(os.path.join(args.frames, 'det_viz_*.png')))[:3]}")


if __name__ == "__main__":
    main()
