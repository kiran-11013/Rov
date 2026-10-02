"""One command = one episode: the drone, holding memory from a mapping pass at t0, is told at t0 + dt to
go to object #oid. Each arm chooses its first viewpoint, then the shared executor flies, observes,
re-identifies and keeps searching until it accepts an object or runs out of budget.

Success (instance-exact): the accepted object is the commanded one and the drone reached its
approach point (within 1 m) inside the time budget. Success (intent-equivalent): it reached an object
of the commanded class (relevant for identical chairs).
"""
from dataclasses import dataclass

import numpy as np

from .belief import make_belief
from .dynamics import ABSENT
from .perception import observe, reidentify
from .policy import Planner


@dataclass
class Command:
    idx: int
    oid: int
    t0: float
    dt: float
    mapped_pid: int
    true_pid: int

    @property
    def moved(self):
        return self.true_pid != self.mapped_pid


@dataclass
class Result:
    arm: str
    cmd: int
    success: bool
    success_intent: bool
    time_s: float
    energy_j: float
    wasted_looks: int
    first_action: str


@dataclass
class Arm:
    name: str
    prior: object          # Prior with .predict(cls, t0, dt); ignored by the oracle
    first: str             # 'trust' | 'verify' | 'ours' | 'search' | 'oracle'
    fixed_height: bool = False
    spatial_reid: bool = True  # use the belief as a spatial prior when re-identifying
    altitudes: tuple = None    # explicit altitude set (overrides fixed_height); None = all configured altitudes


class Executor:
    def __init__(self, drone, world, cfg, displacement, absent_model):
        self.d, self.world, self.cfg = drone, world, cfg
        self.disp = displacement
        self.absent_model = absent_model  # prior exposing absent_given_moved(cls, dt), shared by all arms
        self._planners = {}

    def planner_for(self, altitudes):
        """Planner restricted to the given altitudes; a single altitude is also the cruise/approach height."""
        key = tuple(sorted(altitudes))
        if key not in self._planners:
            approach = key[0] if len(key) == 1 else self.cfg.cruise_altitude
            self._planners[key] = Planner(self.d, key, approach)
        return self._planners[key]

    def arm_altitudes(self, arm):
        if arm.altitudes is not None:
            return tuple(arm.altitudes)
        return (self.cfg.fixed_altitude,) if arm.fixed_height else tuple(self.cfg.altitudes)

    def belief(self, prior, cmd):
        cls = self.world.objects[cmd.oid].cls
        p = float(np.clip(prior.predict(cls, cmd.t0, cmd.dt), 1e-4, 1 - 1e-4))
        q_abs = self.absent_model.absent_given_moved(cls, cmd.dt)
        return make_belief(self.world, self.disp, cls, cmd.mapped_pid, p, q_abs)

    def run(self, arm, cmd, state_now, rng):
        cfg, d = self.cfg, self.d
        alts = self.arm_altitudes(arm)
        pl = self.planner_for(alts)
        alt0 = alts[0] if len(alts) == 1 else cfg.cruise_altitude
        cur = d.dock_vp(alt0)
        target = self.world.objects[cmd.oid]

        if arm.first == "oracle":
            app = pl.app[cmd.true_pid]
            t = d.TIME[cur, app]
            return Result(arm.name, cmd.idx, t <= cfg.time_budget_s, t <= cfg.time_budget_s, float(t),
                          float(d.ENERGY[cur, app]), 0, "oracle")

        b = self.belief(arm.prior, cmd)
        if arm.first == "trust":
            action, vp = "trust", pl.trust_vp(cmd.mapped_pid)
        elif arm.first == "verify":
            action, vp = "verify", pl.best_verify_vp(cmd.mapped_pid, cur)
        elif arm.first == "search":
            action, vp = "search", pl.next_vp(b[:pl.P], cur)
        else:
            action, vp, _ = pl.decide(b, cmd.mapped_pid, cur)

        m = b[:pl.P].copy()
        comp = self.world.compatible_places(target.cls)
        t = e = 0.0
        wasted = 0
        while vp is not None:
            dt = d.TIME[cur, vp] + d.obs_time
            if t + dt > cfg.time_budget_s:
                break
            t += dt
            e += d.ENERGY[cur, vp] + d.obs_energy
            cur = vp
            dets = observe(d, self.world, state_now, vp, rng, cls_filter=target.cls)
            prior_here = m / m.sum() if (arm.spatial_reid and m.sum() > 0) else None
            hit = reidentify(dets, target.appearance, cfg.reid_threshold, belief=prior_here)
            if hit is not None:
                app = pl.app[hit.pid]
                t_fin = t + d.TIME[cur, app]
                if t_fin > cfg.time_budget_s:
                    break
                return Result(arm.name, cmd.idx, hit.oid == cmd.oid, True, float(t_fin),
                              float(e + d.ENERGY[cur, app]), wasted, action)
            wasted += 1
            m *= 1.0 - d.V[vp]
            if m.sum() < 1e-6:  # belief exhausted: fall back to a sweep over all compatible places
                m = np.zeros(pl.P)
                m[comp] = 1.0 / len(comp)
            vp = pl.next_vp(m, cur)
        return Result(arm.name, cmd.idx, False, False, float(cfg.time_budget_s), float(e), wasted, action)


def sample_commands(history, world, cfg, rng, t_start, t_end, n_per_horizon):
    """Commands at t0 + dt (dt in cfg.horizons_h); t0 and t0 + dt both in working hours on weekdays;
    the target is present at both times (the user asks for something that exists)."""
    cmds = []
    for dt in cfg.horizons_h:
        tries = 0
        made = 0
        while made < n_per_horizon and tries < n_per_horizon * 200:
            tries += 1
            day = int(rng.integers(int(t_start // 24), int(t_end // 24)))
            t0 = day * 24 + float(rng.uniform(9.5, 17.5))
            t1 = t0 + dt
            if t1 >= t_end or (day % 7) >= 5 or (int(t1 // 24) % 7) >= 5 or not (9.0 <= t1 % 24 < 19.0):
                continue
            oid = int(rng.integers(len(world.objects)))
            p0, p1 = history.place_at(oid, t0), history.place_at(oid, t1)
            if p0 == ABSENT or p1 == ABSENT:
                continue
            cmds.append(Command(len(cmds), oid, t0, dt, p0, p1))
            made += 1
    return cmds
