"""Stage 3b: can appearance (DINOv2) and metric size (depth) decide *which* object a detection is?

Uses the frames and raw detections from capture.py (--rgb-every 1 --save-seg) and detect_frames.py:
    ~/isaacsim/python.sh isaac/reid_frames.py --spec S --frames F

Memory: for every object, a crop from the frame where it shows the most pixels (the mapping view).
Queries: every detection box (confidence >= --conf) that overlaps exactly one object tightly, from viewpoints
other than the object's memory viewpoint. For each query we embed the crop with DINOv2 and compare it with all
memories (cosine similarity), then report
    * retrieval: is the most similar memory the right object? (among all objects / among its own class)
    * separation (ROC AUC) of the right memory vs other-class memories and vs same-class look-alikes
And a size check: the box's metric height from depth (box height in px x depth / focal) against each class's
size, to see whether size rejects the labels the detector confuses (trash can -> "cup", stool -> "chair").
Writes <frames>/REID.md.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detect_frames as DF  # noqa: E402

# class heights (m) of the placed models (see the [scene] model lines), for the size check
CLASS_HEIGHT = {"chair": 0.81, "stool": 0.45, "cart": 0.90, "bag": 0.11, "laptop": 0.01, "monitor": 0.31,
                "box": 0.12, "bin": 0.40, "plant": 0.27, "toolbox": 0.20, "mug": 0.07}
# for laptops the visible extent is the lid, not the 1 cm height; compare the largest side instead
CLASS_EXTENT = {"chair": 0.81, "stool": 0.45, "cart": 0.90, "bag": 0.22, "laptop": 0.28, "monitor": 0.55,
                "box": 0.40, "bin": 0.40, "plant": 0.27, "toolbox": 0.24, "mug": 0.09}


def auc(pos, neg):
    """P(score of a positive > score of a negative), ties count half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def crop(img, box, pad=0.1):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, y0 = max(0, x0 - pad * w), max(0, y0 - pad * h)
    x1, y1 = min(img.width, x1 + pad * w), min(img.height, y1 + pad * h)
    return img.crop((int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))))


def metric_extent(depth, box, focal):
    """(height, largest side) in metres of a box, from the 30th-percentile depth inside its central region."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    w, h = x1 - x0, y1 - y0
    inner = depth[y0 + h // 4:y1 - h // 4 + 1, x0 + w // 4:x1 - w // 4 + 1].astype(np.float32)
    inner = inner[np.isfinite(inner) & (inner > 0.05)]
    if inner.size == 0:
        return None
    z = float(np.percentile(inner, 30))
    return h * z / focal, max(w, h) * z / focal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--max-queries", type=int, default=3000)
    ap.add_argument("--size-tol", type=float, default=2.0, help="size check: accept a class if within this factor")
    args = ap.parse_args()

    with open(args.spec) as fh:
        spec = json.load(fh)
    with open(os.path.join(args.frames, "detections_raw.json")) as fh:
        raw = json.load(fh)["frames"]
    cls_of = {o["oid"]: o["cls"] for o in spec["objects"]}
    cam = spec["camera"]
    focal = (cam["width"] / 2) / math.tan(math.radians(cam["hfov_deg"]) / 2)

    # memory views: the frame where each object shows the most pixels
    best = {}
    for fr in raw:
        for o, (px, box) in fr["gt"].items():
            if px > best.get(int(o), (0,))[0]:
                best[int(o)] = (px, box, fr["frame"], fr["node"])
    # queries: detection boxes that overlap exactly one object tightly, away from that object's memory viewpoint
    queries = []
    for fr in raw:
        ov = DF.tight_overlaps(fr)
        owners = defaultdict(list)
        for oid, lst in ov.items():
            for j, sc in lst:
                owners[j].append((sc, oid))
        for j, own in owners.items():
            det = fr["dets"][j]
            if det[1] < args.conf or len(own) != 1:
                continue
            oid = own[0][1]
            if oid in best and best[oid][3] != fr["node"]:
                queries.append({"frame": fr["frame"], "oid": oid, "box": det[2], "label": det[0], "conf": det[1]})
    rng = np.random.default_rng(0)
    if len(queries) > args.max_queries:
        queries = [queries[i] for i in sorted(rng.choice(len(queries), args.max_queries, replace=False))]
    print(f"[reid] {len(best)} memories, {len(queries)} queries, model {args.model}")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", args.model).to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def embed(images):
        out = []
        for k in range(0, len(images), 64):
            x = np.stack([np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC), np.float32) / 255
                          for im in images[k:k + 64]])
            t = (torch.from_numpy(x).permute(0, 3, 1, 2).to(dev) - mean) / std
            with torch.no_grad():
                f = model(t)
            out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(out)

    def rgb(frame):
        return Image.open(os.path.join(args.frames, f"rgb_{frame:05d}.png"))

    mem_ids = sorted(best)
    M = embed([crop(rgb(best[o][2]), best[o][1]) for o in mem_ids])
    by_frame = defaultdict(list)
    for qi, q in enumerate(queries):
        by_frame[q["frame"]].append(qi)
    crops, sizes = [None] * len(queries), [None] * len(queries)
    for frame, qis in by_frame.items():
        im = rgb(frame)
        z = np.load(os.path.join(args.frames, f"seg_{frame:05d}.npz"))
        depth = z["depth"]
        for qi in qis:
            crops[qi] = crop(im, queries[qi]["box"])
            sizes[qi] = metric_extent(depth, queries[qi]["box"], focal)
    Q = embed(crops)
    S = Q @ M.T  # queries x memories
    col = {o: i for i, o in enumerate(mem_ids)}

    top1_all, top1_cls, pos, neg_other, neg_same = [], [], [], [], []
    per_class = defaultdict(lambda: [0, 0, 0])
    for qi, q in enumerate(queries):
        o, c = q["oid"], cls_of[q["oid"]]
        s = S[qi]
        same = [col[x] for x in mem_ids if cls_of[x] == c and x != o]
        other = [col[x] for x in mem_ids if cls_of[x] != c]
        top1_all.append(mem_ids[int(np.argmax(s))] == o)
        cls_idx = [col[x] for x in mem_ids if cls_of[x] == c]
        top1_cls.append(mem_ids[cls_idx[int(np.argmax(s[cls_idx]))]] == o)
        pos.append(s[col[o]])
        neg_other.extend(s[other].tolist())
        neg_same.extend(s[same].tolist())
        pc = per_class[c]
        pc[0] += 1
        pc[1] += top1_all[-1]
        pc[2] += mem_ids[int(np.argmax(s))] in [x for x in mem_ids if cls_of[x] == c]

    L = ["# Stage 3b: re-identification with DINOv2 and a depth size check\n",
         f"Layout `{spec['meta']['layout']}`. Memory = crop from each object's clearest frame; queries = {len(queries)} "
         f"detection boxes (confidence ≥ {args.conf}) from other viewpoints, each overlapping exactly one object. "
         f"Embedding `{args.model}`, cosine similarity.\n",
         f"- Most similar memory is the right object: **{np.mean(top1_all):.1%}** (chance {1 / len(mem_ids):.1%})",
         f"- Most similar memory is the right class: **{sum(v[2] for v in per_class.values()) / len(queries):.1%}**",
         f"- Among memories of its own class, the right object is most similar: **{np.mean(top1_cls):.1%}**",
         f"- Separation, right object vs other classes: AUC **{auc(pos, neg_other):.3f}**; vs same-class look-alikes: "
         f"AUC **{auc(pos, neg_same):.3f}** (0.5 = indistinguishable)\n",
         "| Class | Queries | Top-1 right object | Top-1 right class | Instances of this class |", "|---|---|---|---|---|"]
    n_inst = defaultdict(int)
    for o in mem_ids:
        n_inst[cls_of[o]] += 1
    for c in sorted(per_class):
        n, a, b = per_class[c]
        L.append(f"| {c} | {n} | {a / n:.1%} | {b / n:.1%} | {n_inst[c]} |")

    # size check
    L += ["\n**Size check from depth.** A label is *plausible* if the box's metric size is within a factor "
          f"{args.size_tol:g} of that class's size (height, or largest side for flat classes).\n",
          "| True class | Detector label | Queries | Right class plausible | Detector's label plausible |",
          "|---|---|---|---|---|"]
    conf_pairs = defaultdict(lambda: [0, 0, 0])

    def plausible(cls, sz):
        h, ext = sz
        ref, val = (CLASS_EXTENT[cls], ext) if cls in ("laptop", "monitor", "box", "bag", "toolbox", "mug") else (CLASS_HEIGHT[cls], h)
        return 1 / args.size_tol <= val / ref <= args.size_tol

    for qi, q in enumerate(queries):
        if sizes[qi] is None:
            continue
        c = cls_of[q["oid"]]
        rec = conf_pairs[(c, q["label"])]
        rec[0] += 1
        rec[1] += plausible(c, sizes[qi])
        rec[2] += plausible(q["label"], sizes[qi])
    for (c, lab), (n, r, w) in sorted(conf_pairs.items(), key=lambda kv: (kv[0][0], -kv[1][0])):
        if n >= 5:
            L.append(f"| {c} | {lab} | {n} | {r / n:.0%} | {w / n:.0%} |")
    wrong = [(n, r, w) for (c, lab), (n, r, w) in conf_pairs.items() if c != lab]
    right = [(n, r, w) for (c, lab), (n, r, w) in conf_pairs.items() if c == lab]
    if wrong and right:
        L.append(f"\nOver all queries: correct labels pass the size check {sum(r for _, r, _ in right) / sum(n for n, _, _ in right):.0%} "
                 f"of the time; wrong labels pass {sum(w for _, _, w in wrong) / sum(n for n, _, _ in wrong):.0%} "
                 f"(lower = the size check rejects more confusions).\n")
    text = "\n".join(L) + "\n"
    with open(os.path.join(args.frames, "REID.md"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
