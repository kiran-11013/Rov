"""Stage 3b (replay): export one command episode, flown by several arms, for rendering in Isaac Sim.

Runs with ordinary Python. Writes a JSON holding the scene at the moment of the command (same format as
export_scene.py), the stale map position and the true position of the target, and for each arm its timed
flight path (see sim/lbyf/replay.py), its look / approach events and its result. `fly_episode.py` renders it.

    python isaac/export_episode.py --layout open --out isaac/out/episode.json            # picks a telling command
    python isaac/export_episode.py --layout open --cmd 17 --out isaac/out/episode.json   # a specific one
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)

import export_scene as ES  # noqa: E402
from lbyf.config import SimConfig  # noqa: E402
from lbyf.episode import Arm, Executor, sample_commands  # noqa: E402
from lbyf.experiments import setup  # noqa: E402
from lbyf.replay import flight_path  # noqa: E402


def run_arms(ctx, ex, arms, cmd):
    state_now = ctx.history.state_at(cmd.t0 + cmd.dt)
    out = []
    for a in arms:
        trace = []
        r = ex.run(a, cmd, state_now, np.random.default_rng((ctx.cfg.seed, cmd.idx)), trace=trace)
        out.append((a, r, trace))
    return out


def pick_command(ctx, ex, arms, cmds, max_tries=400):
    """A moved, non-chair target that 'ours' finds, preferring the largest time saving over the first arm."""
    best, best_gain = None, -np.inf
    for cmd in cmds[:max_tries]:
        if not cmd.moved or ctx.world.objects[cmd.oid].cls == "chair":
            continue
        runs = run_arms(ctx, ex, arms, cmd)
        base, ours = runs[0][1], runs[-1][1]
        if not ours.success or ours.time_s > 120:
            continue
        gain = base.time_s - ours.time_s
        if gain > best_gain:
            best, best_gain = cmd, gain
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="open", choices=("open", "cubicle", "booth"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visibility", default="extended", choices=("point", "extended"))
    ap.add_argument("--a50", type=float, default=224.5)
    ap.add_argument("--laptop-open", action="store_true")
    ap.add_argument("--cmd", type=int, default=None, help="command index (default: pick a telling one)")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "episode.json"))
    args = ap.parse_args()

    cfg = SimConfig(seed=args.seed, layout=args.layout, visibility_model=args.visibility, vis_a50_px=args.a50,
                    laptop_open=args.laptop_open)
    ctx = setup(cfg)
    ex = Executor(ctx.drone, ctx.world, cfg, ctx.disp, ctx.ours)
    arms = [Arm("Trust-then-search", ctx.ours, "trust"), Arm("Ours @0.4 m only", ctx.ours, "ours", fixed_height=True),
            Arm("Ours", ctx.ours, "ours")]
    cmds = sample_commands(ctx.history, ctx.world, cfg, np.random.default_rng(cfg.seed + 2), ctx.train_end,
                           ctx.test_end, cfg.n_commands_per_horizon)
    cmd = cmds[args.cmd] if args.cmd is not None else pick_command(ctx, ex, arms, cmds)
    if cmd is None:
        raise SystemExit("no suitable command found; pass --cmd")

    t_now = cmd.t0 + cmd.dt
    spec = ES.build_spec(cfg, t=t_now, n_nodes=0, world=ctx.world, history=ctx.history, drone=ctx.drone)
    place = lambda pid: {"pid": int(pid), "x": ctx.world.places[pid].x, "y": ctx.world.places[pid].y,  # noqa: E731
                         "z": ctx.world.places[pid].z}
    tgt = ctx.world.objects[cmd.oid]
    episode = {"cmd": {"idx": cmd.idx, "oid": cmd.oid, "cls": tgt.cls, "colour": tgt.colour,
                       "t_map_h": cmd.t0, "t_now_h": t_now, "dt_h": cmd.dt, "moved": cmd.moved,
                       "mapped": place(cmd.mapped_pid), "true": place(cmd.true_pid)},
               "arms": []}
    for a, r, trace in run_arms(ctx, ex, arms, cmd):
        keys, events = flight_path(ctx.drone, trace)
        episode["arms"].append({"name": a.name, "success": bool(r.success), "time_s": r.time_s,
                                "energy_j": r.energy_j, "first_action": r.first_action,
                                "keys": [list(k) for k in keys], "events": events})
    spec["episode"] = episode
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(spec, fh, indent=1)
    print(f"command {cmd.idx}: go to #{cmd.oid} ({tgt.colour} {tgt.cls}), mapped {cmd.dt:g} h ago, "
          f"{'MOVED' if cmd.moved else 'not moved'}")
    for arm in episode["arms"]:
        looks = sum(e["kind"] == "look" for e in arm["events"])
        print(f"  {arm['name']:<20} first={arm['first_action']:<7} looks={looks:<2} time={arm['time_s']:6.1f} s "
              f"success={arm['success']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
