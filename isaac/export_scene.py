"""Export one moment of the Stage 1/2 simulated world as a JSON scene spec for Isaac Sim.

Runs with ordinary Python (no Isaac Sim needed). The spec holds the room geometry, the objects present at
time t with simple proxy shapes, the camera viewpoints, and the *model* detection probability
(line-of-sight + range, as used in Stages 1-2) for every (viewpoint, object) pair. `capture.py` renders the
same viewpoints in Isaac Sim and `analyze_visibility.py` compares the two.

    python isaac/export_scene.py --layout booth --out isaac/out/scene_spec.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))

from lbyf.config import SimConfig  # noqa: E402
from lbyf.drone import DroneModel  # noqa: E402
from lbyf.visibility import CLASS_DIMS  # noqa: E402
from lbyf.dynamics import ABSENT, simulate_history  # noqa: E402
from lbyf.world import make_world  # noqa: E402

# Proxy geometry per class lives in the simulator, so the render and the model use the same sizes.
CLASS_SHAPES = CLASS_DIMS
WALL_HEIGHT = 2.6


def default_time(cfg):
    """A weekday at 11:00 in the test weeks (after training)."""
    day = cfg.train_days + 1
    while day % 7 >= 5:
        day += 1
    return day * 24 + 11.0


def build_spec(cfg, t=None, n_nodes=30, yaws=(0, 90, 180, 270), sample_seed=0):
    world = make_world(cfg.seed, **cfg.world_kwargs())
    history = simulate_history(world, cfg, np.random.default_rng(cfg.seed))
    drone = DroneModel(world, cfg)
    t = default_time(cfg) if t is None else t
    state = history.state_at(t)

    objects = []
    for oid, pid in sorted(state.items()):
        if pid == ABSENT:
            continue
        o, p = world.objects[oid], world.places[pid]
        shape, dims = CLASS_SHAPES[o.cls]
        objects.append({"oid": oid, "cls": o.cls, "colour": o.colour, "pid": pid, "x": p.x, "y": p.y,
                        "z_base": p.z, "kind": p.kind, "shape": shape, "dims": list(dims),
                        "prim": f"/World/Objects/obj_{oid}"})

    vantage = np.where(drone.is_vantage)[0]
    rng = np.random.default_rng(sample_seed)
    nodes = sorted(rng.choice(vantage, size=min(n_nodes, len(vantage)), replace=False).tolist())
    viewpoints = []
    for node in nodes:
        x, y = (float(v) for v in drone.node_xy[node])
        for alt in cfg.altitudes:
            vp = drone.vp(node, alt)
            viewpoints.append({"node": int(node), "vp": int(vp), "x": x, "y": y, "alt": float(alt),
                               "yaws": list(yaws),
                               "model_p": {str(ob["oid"]): float(drone.V[vp, ob["pid"]]) for ob in objects}})

    walls = [{"a": list(a), "b": list(b)} for a, b in world.walls]
    outer = [((0, 0), (world.width, 0)), ((world.width, 0), (world.width, world.height)),
             ((world.width, world.height), (0, world.height)), ((0, world.height), (0, 0))]
    walls += [{"a": list(a), "b": list(b), "outer": True} for a, b in outer]
    boxes = [{"name": b.name, "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1, "h": b.h} for b in world.boxes]

    return {
        "meta": {"seed": cfg.seed, "layout": cfg.layout, "partition_h": cfg.partition_h, "clutter_h": cfg.clutter_h,
                 "time_h": t, "weekday": int(t // 24) % 7, "hour": t % 24,
                 "model_object_height": cfg.object_height, "model_max_range": cfg.max_range},
        "room": {"width": world.width, "height": world.height, "wall_height": WALL_HEIGHT},
        "walls": walls, "boxes": boxes, "objects": objects, "viewpoints": viewpoints,
        "camera": {"width": 1280, "height": 720, "hfov_deg": 110.0, "pitch_down_deg": 15.0,
                   "near": 0.05, "far": 30.0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="booth", choices=("open", "cubicle", "booth"))
    ap.add_argument("--partition-h", type=float, default=None)
    ap.add_argument("--clutter-h", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-h", type=float, default=None)
    ap.add_argument("--nodes", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "scene_spec.json"))
    args = ap.parse_args()
    cfg = SimConfig(seed=args.seed, layout=args.layout, partition_h=args.partition_h, clutter_h=args.clutter_h)
    spec = build_spec(cfg, args.time_h, args.nodes)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(spec, fh, indent=1)
    n_frames = sum(len(v["yaws"]) for v in spec["viewpoints"])
    print(f"wrote {args.out}: {len(spec['objects'])} objects, {len(spec['viewpoints'])} viewpoints, "
          f"{n_frames} frames to render")


if __name__ == "__main__":
    main()
