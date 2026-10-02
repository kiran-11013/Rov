"""Turn an episode trace (Executor.run(..., trace=[])) into a timed 3D flight path for rendering.

Timing matches the simulator's cost model: TIME = horizontal path length / v_xy + |climb| / v_z, so the drone
changes altitude first, then flies the grid path the planner costed, then hovers for observe_time_s while
turning a full circle (the 4 yaws the visibility model assumes). Keyframes are (t, x, y, z, yaw_deg) with yaw
unwrapped, so linear interpolation between keyframes is the replay.
"""
import math

import numpy as np


def _closest_yaw(target, prev):
    """target (deg) shifted by multiples of 360 to be nearest to prev."""
    return target + 360.0 * round((prev - target) / 360.0)


def flight_path(drone, trace):
    cfg = drone.cfg
    keys = []

    def add(t, x, y, z, yaw):
        if keys and abs(keys[-1][0] - t) < 1e-9:
            keys[-1] = (t, x, y, z, yaw)
        else:
            keys.append((t, x, y, z, yaw))

    def vp_pose(vp):
        x, y = drone.node_xy[drone.vp_node[vp]]
        return float(x), float(y), float(drone.vp_alt[vp])

    def fly(t, cur, nxt, yaw):
        """Fly cur -> nxt starting at t; returns (arrival time, heading)."""
        x0, y0, z0 = vp_pose(cur)
        x1, y1, z1 = vp_pose(nxt)
        if abs(z1 - z0) > 1e-9:
            t += abs(z1 - z0) / cfg.v_z
            add(t, x0, y0, z1, yaw)
        n0, n1 = drone.nodes[drone.vp_node[cur]], drone.nodes[drone.vp_node[nxt]]
        cells = drone.grid.path(n0, n1) if n0 != n1 else [n0]
        pts = [drone.grid.center(c) for c in cells]
        pts[0], pts[-1] = (x0, y0), (x1, y1)
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            seg = math.hypot(bx - ax, by - ay)
            if seg < 1e-9:
                continue
            yaw = _closest_yaw(math.degrees(math.atan2(by - ay, bx - ax)), yaw)
            add(t, ax, ay, z1, yaw)  # turn on the spot (instantaneous) before each straight segment
            t += seg / cfg.v_xy
            add(t, bx, by, z1, yaw)
        return t, yaw

    start = trace[0]
    assert start["kind"] == "start"
    cur, yaw, t = start["vp"], 0.0, 0.0
    add(0.0, *vp_pose(cur), yaw)
    events = []
    for ev in trace[1:]:
        if ev["kind"] == "look":
            t_arrive, yaw = fly(t, cur, ev["vp"], yaw)
            x, y, z = vp_pose(ev["vp"])
            for k in range(1, 5):  # observe: one full turn
                add(t_arrive + cfg.observe_time_s * k / 4, x, y, z, yaw + 90.0 * k)
            yaw += 360.0
            t, cur = t_arrive + cfg.observe_time_s, ev["vp"]
            events.append({**ev, "t_arrive": t_arrive, "t_end": t, "x": x, "y": y, "z": z})
        elif ev["kind"] == "approach":
            t_arrive, yaw = fly(t, cur, ev["vp"], yaw)
            t, cur = t_arrive, ev["vp"]
            events.append({**ev, "t_arrive": t_arrive, "t_end": t, **dict(zip("xyz", vp_pose(cur)))})
    return keys, events


def pose_at(keys, t):
    """Linear interpolation of the keyframes at time t (clamped)."""
    ts = np.array([k[0] for k in keys])
    if t <= ts[0]:
        return keys[0][1:]
    if t >= ts[-1]:
        return keys[-1][1:]
    i = int(np.searchsorted(ts, t, side="right")) - 1
    (t0, *a), (t1, *b) = keys[i], keys[i + 1]
    f = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
    return tuple(av + f * (bv - av) for av, bv in zip(a, b))
