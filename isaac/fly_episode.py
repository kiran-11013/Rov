"""Stage 3b (replay): fly an exported episode in Isaac Sim and render it as a video.

Run with Isaac Sim's own Python (tested target: Isaac Sim 6.0.1, Ubuntu 24.04):

    python3 isaac/export_episode.py --layout open --out isaac/out/episode.json
    ~/isaacsim/python.sh isaac/fly_episode.py --episode isaac/out/episode.json --out isaac/out/fly --headless

The drone is built from plain USD shapes and follows the timed path from the simulator (kinematic replay,
no flight physics yet; that is Stage 3c). Each video frame is two panels:
    left   overview camera straight above the room, with the flight trail, the stale map position (red ring)
           and every look (blue = nothing accepted, green = target accepted)
    right  the onboard ZED-like camera (1280x720, 110 deg HFOV, 15 deg down); pixels of the commanded object
           are tinted green, so you can see when the target is actually in view
Writes <out>/<arm>/frame_XXXXX.png and, if ffmpeg is installed, <out>/<arm>.mp4.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--headless", action="store_true")
ap.add_argument("--arms", default="all", help="comma-separated arm indices, e.g. 0,2 (default all)")
ap.add_argument("--fps", type=float, default=10.0, help="video frames per second")
ap.add_argument("--speed", type=float, default=2.0, help="simulated seconds per video second")
ap.add_argument("--rt-subframes", type=int, default=2)
ap.add_argument("--max-frames", type=int, default=0, help="per arm, 0 = all (use e.g. 20 for a quick test)")
args = ap.parse_args()

with open(args.episode) as fh:
    SPEC = json.load(fh)
EP = SPEC["episode"]
CAM = SPEC["camera"]
OVER = {"width": 1280, "height": 720, "hfov_deg": 90.0, "z": 11.0}

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": CAM["width"], "height": CAM["height"]})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
from lbyf.replay import pose_at  # noqa: E402

COLOURS = {"red": (0.75, 0.12, 0.10), "blue": (0.12, 0.30, 0.75), "green": (0.12, 0.55, 0.20),
           "black": (0.05, 0.05, 0.05), "white": (0.92, 0.92, 0.90), "grey": (0.50, 0.50, 0.50),
           "yellow": (0.90, 0.75, 0.10)}
FURNITURE = {"desk": (0.62, 0.50, 0.35), "table": (0.62, 0.50, 0.35), "part": (0.55, 0.62, 0.72),
             "cub": (0.55, 0.62, 0.72), "shelf": (0.35, 0.35, 0.38), "clutter": (0.30, 0.25, 0.22)}
OBJ_RE = re.compile(r"/World/Objects/obj_(\d+)")
CAM_FORWARD = 0.12  # onboard camera sits this far ahead of the drone centre, clear of the frame


def log(msg):
    print(f"[fly] {msg}", flush=True)


# ----------------------------------------------------------------------------- scene (same as capture.py)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")


def cube(path, cx, cy, cz, sx, sy, sz, colour, rot_z=0.0):
    c = UsdGeom.Cube.Define(stage, path)
    c.GetSizeAttr().Set(1.0)
    xf = UsdGeom.XformCommonAPI(c)
    xf.SetTranslate(Gf.Vec3d(cx, cy, cz))
    if rot_z:
        xf.SetRotate(Gf.Vec3f(0.0, 0.0, rot_z), UsdGeom.XformCommonAPI.RotationOrderXYZ)
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
for ob in SPEC["objects"]:
    colour = COLOURS.get(ob["colour"], (0.5, 0.5, 0.5))
    if ob["shape"] == "box":
        sx, sy, sz = ob["dims"]
        cube(ob["prim"], ob["x"], ob["y"], ob["z_base"] + sz / 2, sx, sy, sz, colour)
    else:
        r, hgt = ob["dims"]
        cylinder(ob["prim"], ob["x"], ob["y"], ob["z_base"], r, hgt, colour)

dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
dome.CreateIntensityAttr(800.0)
sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
sun.CreateIntensityAttr(2500.0)
UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-50.0, 20.0, 0.0))

# ----------------------------------------------------------------------------- drone (plain USD, ~0.45 m)
drone = UsdGeom.Xform.Define(stage, "/World/Drone")
drone_xf = UsdGeom.XformCommonAPI(drone)
cube("/World/Drone/body", 0, 0, 0, 0.20, 0.14, 0.07, (0.95, 0.45, 0.05))
cube("/World/Drone/arm_a", 0, 0, 0, 0.46, 0.03, 0.02, (0.15, 0.15, 0.15), rot_z=45.0)
cube("/World/Drone/arm_b", 0, 0, 0, 0.46, 0.03, 0.02, (0.15, 0.15, 0.15), rot_z=-45.0)
for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
    cylinder(f"/World/Drone/rotor_{k}", 0.163 * sx, 0.163 * sy, 0.01, 0.09, 0.012,
             (0.85, 0.10, 0.10) if sx > 0 else (0.20, 0.20, 0.20))  # red rotors at the front
cube("/World/Drone/zed", 0.10, 0, 0.0, 0.03, 0.17, 0.03, (0.10, 0.10, 0.10))

# ----------------------------------------------------------------------------- cameras


def make_camera(path, width, height, hfov_deg, near=0.05, far=30.0):
    cam = UsdGeom.Camera.Define(stage, path)
    h_ap = 20.955
    cam.GetHorizontalApertureAttr().Set(h_ap)
    cam.GetVerticalApertureAttr().Set(h_ap * height / width)
    cam.GetFocalLengthAttr().Set(h_ap / (2 * math.tan(math.radians(hfov_deg) / 2)))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(near, far))
    return cam, UsdGeom.XformCommonAPI(cam)


def set_pose(xf, x, y, z, yaw_deg, pitch_down_deg):
    # Same convention as capture.py: Rx(90 - pitch) points the camera along +Y, tilted down; Rz(yaw - 90)
    # turns it to face `yaw` measured from +X.
    xf.SetTranslate(Gf.Vec3d(x, y, z))
    xf.SetRotate(Gf.Vec3f(90.0 - pitch_down_deg, 0.0, yaw_deg - 90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


onboard, onboard_xf = make_camera("/World/OnboardCam", CAM["width"], CAM["height"], CAM["hfov_deg"],
                                  CAM["near"], CAM["far"])
over, over_xf = make_camera("/World/OverviewCam", OVER["width"], OVER["height"], OVER["hfov_deg"], 0.5, 40.0)
OX, OY = room["width"] / 2, room["height"] / 2
set_pose(over_xf, OX, OY, OVER["z"], 90.0, 90.0)  # straight down, image up = +y, image right = +x
OF = (OVER["width"] / 2) / math.tan(math.radians(OVER["hfov_deg"]) / 2)


def to_overview_px(x, y, z=0.0):
    s = OF / (OVER["z"] - z)
    return OVER["width"] / 2 + s * (x - OX), OVER["height"] / 2 - s * (y - OY)


rp_on = rep.create.render_product(str(onboard.GetPath()), (CAM["width"], CAM["height"]))
rp_over = rep.create.render_product(str(over.GetPath()), (OVER["width"], OVER["height"]))
ann_on = rep.AnnotatorRegistry.get_annotator("rgb")
ann_inst = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
ann_over = rep.AnnotatorRegistry.get_annotator("rgb")
ann_on.attach([rp_on])
ann_inst.attach([rp_on])
ann_over.attach([rp_over])
try:
    rep.orchestrator.set_capture_on_play(False)
except Exception:
    pass


def step():
    try:
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
    except TypeError:
        rep.orchestrator.step()


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_BIG, F_SMALL = font(30), font(22)


def place_drone(x, y, z, yaw):
    drone_xf.SetTranslate(Gf.Vec3d(x, y, z))
    drone_xf.SetRotate(Gf.Vec3f(0.0, 0.0, yaw), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    yr = math.radians(yaw)
    set_pose(onboard_xf, x + CAM_FORWARD * math.cos(yr), y + CAM_FORWARD * math.sin(yr), z, yaw,
             CAM["pitch_down_deg"])


def target_mask(oid):
    seg = ann_inst.get_data()
    ids = np.asarray(seg["data"]).reshape(CAM["height"], CAM["width"])
    info = seg.get("info", {})
    id_to_path = info.get("idToLabels") or info.get("idToSemantics") or {}
    hits = [int(k) for k, v in id_to_path.items() if (m := OBJ_RE.search(str(v))) and int(m.group(1)) == oid]
    return np.isin(ids, hits) if hits else np.zeros(ids.shape, bool)


def phase_at(events, t):
    for e in events:
        if e["kind"] == "look" and e["t_arrive"] <= t <= e["t_end"]:
            return f"LOOK ({e['step']}) at {e['z']:.1f} m"
    nxt = next((e for e in events if e["t_arrive"] > t), None)
    if nxt is None:
        return "ARRIVED"
    return "FLY TO TARGET" if nxt["kind"] == "approach" else "FLY TO NEXT VIEWPOINT"


# ----------------------------------------------------------------------------- render loop
for _ in range(30):  # let materials and lights settle
    app.update()

cmd = EP["cmd"]
oid = cmd["oid"]
header = (f"Go to #{oid} ({cmd['colour']} {cmd['cls']}), mapped {cmd['dt_h']:g} h ago - "
          f"{'it has MOVED' if cmd['moved'] else 'still in place'}")
arm_ids = range(len(EP["arms"])) if args.arms == "all" else [int(a) for a in args.arms.split(",")]
os.makedirs(args.out, exist_ok=True)
summary = []
for ai in arm_ids:
    arm = EP["arms"][ai]
    keys, events = arm["keys"], arm["events"]
    slug = re.sub(r"[^a-z0-9]+", "_", arm["name"].lower()).strip("_")
    out_dir = os.path.join(args.out, f"{ai}_{slug}")
    os.makedirs(out_dir, exist_ok=True)
    t_end = keys[-1][0] + 1.5  # hold the last pose briefly
    n = int(math.ceil(t_end * args.fps / args.speed)) + 1
    if args.max_frames:
        n = min(n, args.max_frames)
    log(f"arm {ai} '{arm['name']}': {arm['time_s']:.1f} s simulated, {len(events)} events, {n} frames")
    place_drone(*pose_at(keys, 0.0))
    for _ in range(3):  # warm-up frames (render vars not ready on the first one)
        step()
    trail, t0 = [], time.time()
    for i in range(n):
        t = min(i * args.speed / args.fps, keys[-1][0])
        x, y, z, yaw = pose_at(keys, t)
        place_drone(x, y, z, yaw)
        step()
        on = np.asarray(ann_on.get_data())[..., :3].astype(np.uint8).copy()
        mask = target_mask(oid)
        if mask.any():
            on[mask] = (0.45 * on[mask] + 0.55 * np.array([40, 230, 60])).astype(np.uint8)
        ov = Image.fromarray(np.asarray(ann_over.get_data())[..., :3].astype(np.uint8).copy())
        d = ImageDraw.Draw(ov)
        trail.append(to_overview_px(x, y, z))
        if len(trail) > 1:
            d.line(trail, fill=(255, 120, 0), width=3)
        mx, my = to_overview_px(cmd["mapped"]["x"], cmd["mapped"]["y"], cmd["mapped"]["z"])
        d.ellipse([mx - 16, my - 16, mx + 16, my + 16], outline=(220, 0, 0), width=4)
        d.text((mx + 18, my - 12), "map says here", fill=(220, 0, 0), font=F_SMALL)
        for e in events:
            if e["kind"] == "look" and e["t_arrive"] <= t:
                lx, ly = to_overview_px(e["x"], e["y"], e["z"])
                col = (30, 200, 60) if e["accepted"] is not None else (40, 110, 255)
                d.ellipse([lx - 8, ly - 8, lx + 8, ly + 8], fill=col)
        d.text((16, 12), f"{arm['name']}   t = {t:5.1f} s   {phase_at(events, t)}", fill=(0, 0, 0), font=F_BIG)
        d.text((16, 50), f"altitude {z:.1f} m", fill=(0, 0, 0), font=F_SMALL)
        oni = Image.fromarray(on)
        d2 = ImageDraw.Draw(oni)
        d2.text((16, 12), "onboard camera" + (f" - target in view ({int(mask.sum())} px)" if mask.any() else ""),
                fill=(255, 255, 255), font=F_BIG)
        frame = Image.new("RGB", (OVER["width"] + CAM["width"], max(OVER["height"], CAM["height"]) + 50), "white")
        frame.paste(ov, (0, 50))
        frame.paste(oni, (OVER["width"], 50))
        ImageDraw.Draw(frame).text((16, 10), header, fill=(0, 0, 0), font=F_BIG)
        frame.save(os.path.join(out_dir, f"frame_{i:05d}.png"))
        if i % 50 == 0 or i == n - 1:
            log(f"  frame {i + 1}/{n} ({(i + 1) / max(time.time() - t0, 1e-6):.1f} fps) t={t:.1f} s "
                f"z={z:.1f} {phase_at(events, t)}")
    video = None
    if shutil.which("ffmpeg"):
        video = os.path.join(args.out, f"{ai}_{slug}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i",
                        os.path.join(out_dir, "frame_%05d.png"), "-vf", "scale=1920:-2", "-pix_fmt", "yuv420p",
                        video], check=False)
    summary.append(f"{arm['name']}: {arm['time_s']:.1f} s, success {arm['success']} -> {video or out_dir}")

for s in summary:
    log(s)
if not shutil.which("ffmpeg"):
    log("ffmpeg not found: frames saved as PNG (sudo apt install ffmpeg, then rerun or join them yourself)")
try:
    rep.orchestrator.wait_until_complete()
except Exception:
    pass
app.close()
sys.exit(0)
