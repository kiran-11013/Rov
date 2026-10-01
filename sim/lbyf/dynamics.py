"""Ground-truth object movement over weeks (the simulated stand-in for E1's tracked logs).

Model:
* Activity: weekdays 09:00-19:00 have activity 1.0, all other times 0.03 (cleaners, late workers).
* Each object's time-to-next-move is Weibull in *activity exposure* (a renewal clock that restarts
  after every move), so objects move mostly during working hours.
* Periodic classes (bags, laptops) leave around 18:00 with probability 0.7 and come back the next
  working morning, usually to the same place.
* A move goes to a free compatible place, in the same room with the class's `same_room` probability.

Time 0 is Monday 00:00. Place id -1 means "not in the mapped space".
"""
import heapq
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

ABSENT = -1


class Activity:
    def __init__(self, days, active=1.0, idle=0.03):
        minutes = np.arange(days * 24 * 60 + 1)
        hours = minutes / 60.0
        day = (hours // 24).astype(int)
        hod = hours % 24
        weekday = (day % 7) < 5
        rate = np.where(weekday & (hod >= 9) & (hod < 19), active, idle)
        self.t = hours
        self.A = np.concatenate([[0.0], np.cumsum(rate[:-1]) / 60.0])
        self.horizon = hours[-1]

    def exposure(self, t):
        """Cumulative activity hours at wall-clock time t (hours)."""
        return np.interp(t, self.t, self.A)

    def wall_time(self, e):
        """Inverse of `exposure`; returns inf beyond the simulated horizon."""
        e = np.asarray(e, dtype=float)
        out = np.interp(e, self.A, self.t)
        return np.where(e > self.A[-1], np.inf, out)


def is_weekday(t):
    return (int(t // 24) % 7) < 5


@dataclass
class History:
    times: dict   # oid -> np.array of record times (sorted)
    places: dict  # oid -> np.array of place ids (ABSENT if away)
    activity: Activity
    horizon: float

    def place_at(self, oid, t):
        i = bisect_right(self.times[oid], t) - 1
        return int(self.places[oid][i]) if i >= 0 else ABSENT

    def state_at(self, t):
        return {oid: self.place_at(oid, t) for oid in self.times}

    def next_change_after(self, oid, t):
        """Time of the first record strictly after t (records only happen on changes)."""
        ts = self.times[oid]
        i = bisect_right(ts, t)
        return float(ts[i]) if i < len(ts) else np.inf

    def moves(self, t_end=np.inf):
        """(oid, time, from_pid, to_pid) for every change before t_end."""
        out = []
        for oid in self.times:
            ts, ps = self.times[oid], self.places[oid]
            for i in range(1, len(ts)):
                if ts[i] < t_end:
                    out.append((oid, float(ts[i]), int(ps[i - 1]), int(ps[i])))
        return out


def simulate_history(world, cfg, rng):
    act = Activity(cfg.days)
    horizon = act.horizon
    n_obj = len(world.objects)
    occupant = {p.pid: None for p in world.places}
    where = np.full(n_obj, ABSENT)
    home = np.full(n_obj, ABSENT)
    times = {o.oid: [0.0] for o in world.objects}
    places = {o.oid: [] for o in world.objects}
    version = np.zeros(n_obj, dtype=int)
    queue, seq = [], 0

    def push(t, oid, kind, ver=None):
        nonlocal seq
        if t < horizon:
            heapq.heappush(queue, (t, seq, oid, kind, ver))
            seq += 1

    def schedule_move(oid, now):
        cl = world.classes[world.objects[oid].cls]
        e = cl.weibull_scale_h * rng.weibull(cl.weibull_k)
        push(float(act.wall_time(act.exposure(now) + e)), oid, "move", version[oid])

    def free_places(cls_name, room=None, exclude=None):
        cand = [pid for pid in world.compatible_places(cls_name) if occupant[pid] is None and pid != exclude]
        if room is not None:
            cand = [pid for pid in cand if world.places[pid].room == room]
        return cand

    def record(oid, t, pid):
        if places[oid] and places[oid][-1] == pid:
            return
        if places[oid]:
            times[oid].append(t)
        places[oid].append(pid)

    def set_place(oid, pid, t):
        old = where[oid]
        if old != ABSENT:
            occupant[old] = None
        if pid != ABSENT:
            occupant[pid] = oid
        where[oid] = pid
        record(oid, t, pid)

    # initial placement: most constrained classes first, random order within a class
    order = sorted(rng.permutation(n_obj), key=lambda o: len(world.compatible_places(world.objects[o].cls)))
    for oid in order:
        cand = free_places(world.objects[oid].cls)
        pid = int(rng.choice(cand))
        home[oid] = pid
        set_place(oid, pid, 0.0)
        schedule_move(oid, 0.0)

    # periodic departures
    for o in world.objects:
        if world.classes[o.cls].periodic:
            for d in range(cfg.days):
                if d % 7 < 5:
                    push(d * 24 + float(np.clip(18 + rng.normal(0, 0.75), 17, 20.5)), o.oid, "leave")

    while queue:
        t, _s, oid, kind, ver = heapq.heappop(queue)
        cl = world.classes[world.objects[oid].cls]
        if kind == "move":
            if ver != version[oid] or where[oid] == ABSENT:
                continue
            room = world.places[where[oid]].room
            if rng.random() < cl.same_room:
                cand = free_places(cl.name, room, exclude=where[oid])
            else:
                other = [r for r in world.rooms if r != room]
                cand = free_places(cl.name, other[rng.integers(len(other))])
            if not cand:
                cand = free_places(cl.name, exclude=where[oid])
            if cand:
                set_place(oid, int(rng.choice(cand)), t)
            schedule_move(oid, t)
        elif kind == "leave":
            if where[oid] == ABSENT or rng.random() > 0.7:
                continue
            home[oid] = where[oid]
            version[oid] += 1  # cancel the pending move
            set_place(oid, ABSENT, t)
            d = int(t // 24) + 1
            while d % 7 >= 5:
                d += 1
            if rng.random() > 0.85:  # occasionally away one extra working day
                d += 1
                while d % 7 >= 5:
                    d += 1
            push(d * 24 + float(np.clip(9.5 + rng.normal(0, 0.5), 8.5, 11.0)), oid, "return")
        elif kind == "return":
            if where[oid] != ABSENT:
                continue
            pid = home[oid]
            if occupant[pid] is not None or rng.random() > 0.85:
                cand = free_places(cl.name, world.places[home[oid]].room) or free_places(cl.name)
                pid = int(rng.choice(cand)) if cand else ABSENT
            if pid != ABSENT:
                set_place(oid, int(pid), t)
                schedule_move(oid, t)

    return History({k: np.array(v) for k, v in times.items()},
                   {k: np.array(v, dtype=int) for k, v in places.items()}, act, horizon)
