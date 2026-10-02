"""Stage 3c: a PX4-controlled drone (Pegasus) flies one of our planner's episodes in the office scene.

Needs PX4-Autopilot v1.16 (built: make px4_sitl_default none) and the Pegasus Simulator extension installed in Isaac
Sim's Python (see isaac/README.md). One command:

    python3 isaac/export_episode.py --layout open --out isaac/out/episode.json      # if not done yet
    ~/isaacsim/python.sh isaac/px4_office.py --episode isaac/out/episode.json --arm 2

What happens:
  * our office scene is built (walls, desks and objects; --assets real for library models) with static collision;
  * a Pegasus quadrotor with PX4 SITL (real flight stack, lockstep) is spawned at the episode's start (the dock);
  * a ZED-like camera (1280x720, 110 deg, 15 deg down) on the body saves frames to --out every --img-every seconds;
  * a MAVLink thread switches PX4 to OFFBOARD, arms, and flies the planner's waypoints (sim/lbyf/replay.py keys)
    with PX4 speed limits set to the simulator's (1 m/s horizontal, 0.5 m/s vertical); at each look it turns a full
    circle, as in the planner. When the last waypoint is reached it lands.
Logs: [px4office] lines, and <out>/flight_log.json (time, setpoint, PX4 position).
"""
import argparse
import json
import math
import os
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True, help="from isaac/export_episode.py")
ap.add_argument("--arm", type=int, default=2, help="index of the arm to fly (0 trust, 1 ours@0.4, 2 ours)")
ap.add_argument("--assets", default="proxy", choices=("proxy", "real"))
ap.add_argument("--out", default="isaac/out/px4_flight")
ap.add_argument("--img-every", type=float, default=0.5, help="seconds of wall time between saved camera frames")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--accept", type=float, default=0.3, help="waypoint acceptance radius (m)")
args = ap.parse_args()

with open(args.episode) as fh:
    SPEC = json.load(fh)
EP = SPEC["episode"]
ARM = EP["arms"][args.arm]
CAM = SPEC["camera"]

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_usd  # noqa: E402


def log(msg):
    print(f"[px4office] {msg}", flush=True)


# ----------------------------------------------------------------------------- world, scene, drone
timeline = omni.timeline.get_timeline_interface()
pg = PegasusInterface()
pg._world = World(**pg._world_settings)
world = pg.world
stage = omni.usd.get_context().get_stage()
scene_usd.build_scene(stage, SPEC, assets=args.assets, colliders=True)

keys = ARM["keys"]  # (t, x, y, z, yaw_deg) in the scene's frame (x east, y north, z up)
x0, y0 = keys[0][1], keys[0][2]
cfg = MultirotorConfig()
cfg.backends = [PX4MavlinkBackend(PX4MavlinkBackendConfig({
    "vehicle_id": 0, "px4_autolaunch": True, "px4_dir": pg.px4_path,
    "px4_vehicle_model": pg.px4_default_airframe}))]
Multirotor("/World/quadrotor", ROBOTS["Iris"], 0, [x0, y0, 0.07],
           Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(), config=cfg)
log(f"drone spawned at the dock ({x0:.1f}, {y0:.1f}); flying arm '{ARM['name']}' ({len(keys)} waypoints, "
    f"{ARM['time_s']:.1f} s in the planner)")

# ZED-like camera on the body, looking forward and 15 deg down (body frame: x forward, y left, z up)
cam = UsdGeom.Camera.Define(stage, "/World/quadrotor/body/zed")
h_ap = 20.955
cam.GetHorizontalApertureAttr().Set(h_ap)
cam.GetVerticalApertureAttr().Set(h_ap * CAM["height"] / CAM["width"])
cam.GetFocalLengthAttr().Set(h_ap / (2 * math.tan(math.radians(CAM["hfov_deg"]) / 2)))
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 30.0))
cxf = UsdGeom.XformCommonAPI(cam)
cxf.SetTranslate(Gf.Vec3d(0.12, 0.0, 0.02))
cxf.SetRotate(Gf.Vec3f(90.0 - CAM["pitch_down_deg"], 0.0, -90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
rp = rep.create.render_product(str(cam.GetPath()), (CAM["width"], CAM["height"]))
ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
ann_rgb.attach([rp])

world.reset()
os.makedirs(args.out, exist_ok=True)

# ----------------------------------------------------------------------------- MAVLink offboard flight
STATE = {"pos_ned": None, "yaw_ned": None, "done": False, "phase": "connecting", "log": []}


def enu_to_ned(x, y, z):
    """Scene (x east, y north, z up) relative to the spawn point -> PX4 local NED."""
    return (y - y0, x - x0, -z)


def yaw_enu_to_ned(yaw_deg):
    return math.radians(90.0 - yaw_deg)


def flight():
    from pymavlink import mavutil
    mav = mavutil.mavlink_connection("udpin:0.0.0.0:14540")
    log("waiting for PX4 heartbeat on udp 14540 ...")
    mav.wait_heartbeat()
    log(f"PX4 connected (system {mav.target_system})")

    def param(name, value, ptype=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
        mav.mav.param_set_send(mav.target_system, mav.target_component, name.encode(), float(value), ptype)
        time.sleep(0.05)

    # speeds of our simulator; offboard without an RC transmitter
    for name, v in (("MPC_XY_VEL_MAX", 1.0), ("MPC_XY_CRUISE", 1.0), ("MPC_Z_VEL_MAX_UP", 0.5),
                    ("MPC_Z_VEL_MAX_DN", 0.5), ("MPC_YAWRAUTO_MAX", 60.0)):
        param(name, v)
    for name, v in (("COM_RCL_EXCEPT", 4), ("NAV_RCL_ACT", 0), ("NAV_DLL_ACT", 0), ("COM_RC_IN_MODE", 4)):
        param(name, v, mavutil.mavlink.MAV_PARAM_TYPE_INT32)

    def send(n, e, d, yaw):
        mask = 0b0000_1001_1111_1000  # position + yaw only
        mav.mav.set_position_target_local_ned_send(
            0, mav.target_system, mav.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED, mask,
            n, e, d, 0, 0, 0, 0, 0, 0, yaw, 0)

    def poll():
        while True:
            m = mav.recv_match(type=["LOCAL_POSITION_NED", "ATTITUDE"], blocking=False)
            if m is None:
                return
            if m.get_type() == "LOCAL_POSITION_NED":
                STATE["pos_ned"] = (m.x, m.y, m.z)
            else:
                STATE["yaw_ned"] = m.yaw

    first = keys[0]
    n, e, d = enu_to_ned(first[1], first[2], first[3])
    yaw = yaw_enu_to_ned(first[4])
    STATE["phase"] = "arming"
    t_arm = time.time()
    while STATE["pos_ned"] is None or time.time() - t_arm < 2.0:  # stream setpoints before OFFBOARD
        send(n, e, d, yaw)
        poll()
        time.sleep(0.05)
    mav.mav.command_long_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                              mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0, 0, 0, 0, 0)  # 6 = OFFBOARD
    mav.mav.command_long_send(mav.target_system, mav.target_component,
                              mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    log("OFFBOARD + arm sent; taking off to the first waypoint")
    t_start = time.time()
    for i, k in enumerate(keys):
        n, e, d = enu_to_ned(k[1], k[2], k[3])
        yaw = yaw_enu_to_ned(k[4] % 360.0)
        STATE["phase"] = f"waypoint {i + 1}/{len(keys)}"
        t0, dist = time.time(), float("nan")
        while True:
            send(n, e, d, yaw)
            poll()
            p, y = STATE["pos_ned"], STATE["yaw_ned"]
            if p is not None:
                STATE["log"].append([round(time.time() - t_start, 2), [n, e, d], list(p)])
                dist = math.dist(p, (n, e, d))
                yaw_err = abs((y - yaw + math.pi) % (2 * math.pi) - math.pi) if y is not None else 0.0
                if dist < args.accept and yaw_err < math.radians(20):
                    break
            if time.time() - t0 > 30.0:
                log(f"waypoint {i + 1} not reached in 30 s (distance {dist:.2f} m), moving on")
                break
            time.sleep(0.05)
        if i % 10 == 0 or i == len(keys) - 1:
            log(f"reached waypoint {i + 1}/{len(keys)} at t={time.time() - t_start:.1f} s, PX4 NED {STATE['pos_ned']}")
    STATE["phase"] = "landing"
    mav.mav.command_long_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_CMD_NAV_LAND,
                              0, 0, 0, 0, 0, 0, 0, 0)
    log(f"episode flown in {time.time() - t_start:.1f} s wall time (planner: {ARM['time_s']:.1f} s); landing")
    STATE["done"] = True


threading.Thread(target=flight, daemon=True).start()

# ----------------------------------------------------------------------------- simulation loop
timeline.play()
t_img, n_img, t_done = 0.0, 0, None
from PIL import Image  # noqa: E402

while simulation_app.is_running():
    world.step(render=True)
    now = time.time()
    if now - t_img >= args.img_every and STATE["phase"].startswith("waypoint"):
        t_img = now
        data = ann_rgb.get_data()
        if data is not None and np.asarray(data).size:
            Image.fromarray(np.asarray(data)[..., :3].astype(np.uint8)).save(
                os.path.join(args.out, f"zed_{n_img:05d}.png"))
            n_img += 1
    if STATE["done"]:
        t_done = t_done or now
        if now - t_done > 8.0:  # let it land
            break

with open(os.path.join(args.out, "flight_log.json"), "w") as fh:
    json.dump({"arm": ARM["name"], "planner_time_s": ARM["time_s"], "log": STATE["log"]}, fh)
log(f"saved {n_img} camera frames and flight_log.json to {args.out}")
carb.log_warn("px4_office closing")
timeline.stop()
simulation_app.close()
