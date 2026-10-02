"""Stage 3a: render the exported scene from every drone viewpoint in Isaac Sim and count visible pixels per object.

Run with Isaac Sim's own Python (tested target: Isaac Sim 6.0.1, Ubuntu 24.04):

    ~/isaacsim/python.sh isaac/capture.py --spec isaac/out/scene_spec.json --out isaac/out/frames --headless

For each viewpoint and yaw it renders RGB, depth and instance segmentation with a ZED-like wide camera
(1280x720, ~110 deg horizontal FOV, pitched 15 deg down), then writes:
    frames/index.json            one record per frame: viewpoint, altitude, yaw, visible pixels per object id
    frames/rgb_XXXXX.png         every --rgb-every frames (for eyeballing)
    frames/seg_XXXXX.npz         instance ids + id->prim map, only with --save-seg
Geometry is plain USD (walls, furniture, proxy objects coloured by their attribute), so it does not depend on
asset packs. Object prims are /World/Objects/obj_<oid>.
"""
import argparse
import json
import math
import os
import re
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--headless", action="store_true")
ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
ap.add_argument("--rt-subframes", type=int, default=4)
ap.add_argument("--rgb-every", type=int, default=10)
ap.add_argument("--save-seg", action="store_true")
args = ap.parse_args()

with open(args.spec) as fh:
    SPEC = json.load(fh)
CAM = SPEC["camera"]

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": CAM["width"], "height": CAM["height"]})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

COLOURS = {"red": (0.75, 0.12, 0.10), "blue": (0.12, 0.30, 0.75), "green": (0.12, 0.55, 0.20),
           "black": (0.05, 0.05, 0.05), "white": (0.92, 0.92, 0.90), "grey": (0.50, 0.50, 0.50),
           "yellow": (0.90, 0.75, 0.10)}
FURNITURE = {"desk": (0.62, 0.50, 0.35), "table": (0.62, 0.50, 0.35), "part": (0.55, 0.62, 0.72),
             "cub": (0.55, 0.62, 0.72), "shelf": (0.35, 0.35, 0.38), "clutter": (0.30, 0.25, 0.22)}
OBJ_RE = re.compile(r"/World/Objects/obj_(\d+)")


def log(msg):
    print(f"[capture] {msg}", flush=True)


# ----------------------------------------------------------------------------- scene
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")


def cube(path, cx, cy, cz, sx, sy, sz, colour):
    c = UsdGeom.Cube.Define(stage, path)
    c.GetSizeAttr().Set(1.0)
    xf = UsdGeom.XformCommonAPI(c)
    xf.SetTranslate(Gf.Vec3d(cx, cy, cz))
    xf.SetScale(Gf.Vec3f(sx, sy, sz))
    c.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return c


def cylinder(path, cx, cy, z_base, radius, height, colour):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.GetRadiusAttr().Set(radius)
    c.GetHeightAttr().Set(height)
    c.GetAxisAttr().Set("Z")
    UsdGeom.XformCommonAPI(c).SetTranslate(Gf.Vec3d(cx, cy, z_base + height / 2))
    c.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return c


def add_label(prim, label):
    """Semantic class label; the API name changed across Isaac Sim versions, so try each."""
    for attempt in ("add_labels", "add_update_semantics"):
        try:
            mod = __import__("isaacsim.core.utils.semantics", fromlist=[attempt])
            fn = getattr(mod, attempt)
            if attempt == "add_labels":
                fn(prim, labels=[label], instance_name="class")
            else:
                fn(prim, label)
            return True
        except Exception:
            continue
    return False


room = SPEC["room"]
cube("/World/Floor", room["width"] / 2, room["height"] / 2, -0.01, room["width"], room["height"], 0.02,
     (0.80, 0.80, 0.78))
for i, w in enumerate(SPEC["walls"]):
    (ax, ay), (bx, by) = w["a"], w["b"]
    length = math.hypot(bx - ax, by - ay)
    horiz = abs(by - ay) < 1e-9
    cube(f"/World/Walls/wall_{i}", (ax + bx) / 2, (ay + by) / 2, room["wall_height"] / 2,
         length if horiz else 0.1, 0.1 if horiz else length, room["wall_height"], (0.88, 0.87, 0.84))
for b in SPEC["boxes"]:
    colour = next((c for k, c in FURNITURE.items() if b["name"].startswith(k)), (0.6, 0.6, 0.6))
    cube(f"/World/Furniture/{b['name']}", (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, b["h"] / 2,
         b["x1"] - b["x0"], b["y1"] - b["y0"], b["h"], colour)

n_labelled = 0
for ob in SPEC["objects"]:
    path, colour = ob["prim"], COLOURS.get(ob["colour"], (0.5, 0.5, 0.5))
    if ob["shape"] == "box":
        sx, sy, sz = ob["dims"]
        prim = cube(path, ob["x"], ob["y"], ob["z_base"] + sz / 2, sx, sy, sz, colour)
    else:
        r, hgt = ob["dims"]
        prim = cylinder(path, ob["x"], ob["y"], ob["z_base"], r, hgt, colour)
    n_labelled += add_label(prim.GetPrim(), ob["cls"])
log(f"built {len(SPEC['objects'])} objects ({n_labelled} with semantic labels), "
    f"{len(SPEC['boxes'])} furniture boxes, {len(SPEC['walls'])} walls")

dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
dome.CreateIntensityAttr(800.0)
sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
sun.CreateIntensityAttr(2500.0)
UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-50.0, 20.0, 0.0))

# ----------------------------------------------------------------------------- camera
cam = UsdGeom.Camera.Define(stage, "/World/DroneCam")
h_ap = 20.955
cam.GetHorizontalApertureAttr().Set(h_ap)
cam.GetVerticalApertureAttr().Set(h_ap * CAM["height"] / CAM["width"])
cam.GetFocalLengthAttr().Set(h_ap / (2 * math.tan(math.radians(CAM["hfov_deg"]) / 2)))
cam.GetClippingRangeAttr().Set(Gf.Vec2f(CAM["near"], CAM["far"]))
cam_xf = UsdGeom.XformCommonAPI(cam)


def set_pose(x, y, z, yaw_deg, pitch_down_deg):
    # USD cameras look down -Z with +Y up. Rx(90 - pitch) points it forward along +Y, tilted down;
    # Rz(yaw - 90) then turns it to face `yaw` measured from +X.
    cam_xf.SetTranslate(Gf.Vec3d(x, y, z))
    cam_xf.SetRotate(Gf.Vec3f(90.0 - pitch_down_deg, 0.0, yaw_deg - 90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


rp = rep.create.render_product(str(cam.GetPath()), (CAM["width"], CAM["height"]))
ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
ann_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
ann_inst = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
for a in (ann_rgb, ann_depth, ann_inst):
    a.attach([rp])
try:
    rep.orchestrator.set_capture_on_play(False)
except Exception:
    pass


def step():
    try:
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
    except TypeError:
        rep.orchestrator.step()


for _ in range(30):  # let materials and lights settle
    app.update()

# ----------------------------------------------------------------------------- capture loop
os.makedirs(args.out, exist_ok=True)
frames = [(vp, yaw) for vp in SPEC["viewpoints"] for yaw in vp["yaws"]]
if args.max_frames:
    frames = frames[:args.max_frames]
index = {"spec": os.path.abspath(args.spec), "camera": CAM, "frames": []}
t0 = time.time()
for i, (vp, yaw) in enumerate(frames):
    set_pose(vp["x"], vp["y"], vp["alt"], yaw, CAM["pitch_down_deg"])
    step()
    seg = ann_inst.get_data()
    ids = np.asarray(seg["data"]).reshape(CAM["height"], CAM["width"])
    info = seg.get("info", {})
    id_to_path = info.get("idToLabels") or info.get("idToSemantics") or {}
    uniq, counts = np.unique(ids, return_counts=True)
    pixels = {}
    for u, c in zip(uniq.tolist(), counts.tolist()):
        m = OBJ_RE.search(str(id_to_path.get(str(u), id_to_path.get(u, ""))))
        if m:
            pixels[m.group(1)] = pixels.get(m.group(1), 0) + int(c)
    rec = {"frame": i, "node": vp["node"], "vp": vp["vp"], "alt": vp["alt"], "yaw": yaw, "pixels": pixels}
    index["frames"].append(rec)
    if args.rgb_every and i % args.rgb_every == 0:
        from PIL import Image
        rgb = np.asarray(ann_rgb.get_data())[..., :3].astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(args.out, f"rgb_{i:05d}.png"))
    if args.save_seg:
        depth = np.asarray(ann_depth.get_data(), dtype=np.float32)
        np.savez_compressed(os.path.join(args.out, f"seg_{i:05d}.npz"), ids=ids.astype(np.uint32),
                            depth=np.nan_to_num(depth, posinf=0).astype(np.float16),
                            id_to_path=json.dumps(id_to_path))
    if i % 20 == 0 or i == len(frames) - 1:
        rate = (i + 1) / max(time.time() - t0, 1e-6)
        log(f"frame {i + 1}/{len(frames)} ({rate:.1f} fps): {len(pixels)} objects visible")
        with open(os.path.join(args.out, "index.json"), "w") as fh:
            json.dump(index, fh)

with open(os.path.join(args.out, "index.json"), "w") as fh:
    json.dump(index, fh)
if not any(f["pixels"] for f in index["frames"]):
    log("WARNING: no object pixels in any frame - check the instance-segmentation id map (see README)")
log(f"done: {len(frames)} frames in {time.time() - t0:.0f} s -> {args.out}")
try:
    rep.orchestrator.wait_until_complete()
except Exception:
    pass
app.close()
sys.exit(0)
