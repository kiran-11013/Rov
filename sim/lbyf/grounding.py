"""E3: grounding language references to map IDs under stale memory, with conformal clarification.

A command such as "the red bag near the store door" refers to the world *now* (command time); the drone's
map is from t0. Resolvers score every remembered object:
* map-only     does the object satisfy the reference at its mapped place?
* stale-aware  expected satisfaction under the belief (still there / moved elsewhere / absent)

Scores are normalised to P(id). Split-conformal calibration on unambiguous commands gives a threshold;
the prediction set {id : 1 - P(id) <= q} is acted on when it has one element, otherwise the drone asks.
A typed decision model (e.g. Jev) or a local LLM would replace the scoring function; the conformal
wrapper and evaluation stay the same.
"""
from dataclasses import dataclass

import numpy as np

from .belief import make_belief
from .dynamics import ABSENT


@dataclass
class Reference:
    cls: str
    colour: str = None
    room: str = None
    near: str = None

    def text(self):
        parts = ["the"] + ([self.colour] if self.colour else []) + [self.cls]
        if self.room:
            parts.append(f"in the {self.room}")
        if self.near:
            parts.append(f"near the {self.near}")
        return " ".join(parts)


def nearest_landmark(world, pid):
    cache = world.__dict__.setdefault("_nearest_landmark", {})
    if pid not in cache:
        p = world.places[pid]
        name, (x, y) = min(world.landmarks.items(), key=lambda kv: np.hypot(kv[1][0] - p.x, kv[1][1] - p.y))
        cache[pid] = name if np.hypot(x - p.x, y - p.y) <= 3.0 else None
    return cache[pid]


def satisfies(world, oid, pid, ref):
    if pid == ABSENT:
        return False
    o, p = world.objects[oid], world.places[pid]
    if o.cls != ref.cls:
        return False
    if ref.colour and o.colour != ref.colour:
        return False
    if ref.room and p.room != ref.room:
        return False
    if ref.near and nearest_landmark(world, pid) != ref.near:
        return False
    return True


@dataclass
class GroundingCase:
    ref: Reference
    target: int
    truth: list        # all objects satisfying the reference now
    t0: float
    dt: float
    memory: dict       # oid -> mapped pid at t0 (present objects only)

    @property
    def ambiguous(self):
        return len(self.truth) > 1


def sample_cases(history, world, rng, n, t_start, t_end, horizons):
    cases = []
    tries = 0
    while len(cases) < n and tries < n * 100:
        tries += 1
        dt = float(horizons[rng.integers(len(horizons))])
        day = int(rng.integers(int(t_start // 24), int(t_end // 24)))
        t0 = day * 24 + float(rng.uniform(9.5, 17.5))
        t1 = t0 + dt
        if t1 >= t_end or day % 7 >= 5 or int(t1 // 24) % 7 >= 5:
            continue
        now = history.state_at(t1)
        present = [oid for oid, pid in now.items() if pid != ABSENT]
        target = int(rng.choice(present))
        o, pid = world.objects[target], now[target]
        ref = Reference(o.cls)
        attrs = ["colour", "room", "near"]
        rng.shuffle(attrs)
        for a in attrs[: int(rng.integers(1, 3))]:
            if a == "colour":
                ref.colour = o.colour
            elif a == "room":
                ref.room = world.places[pid].room
            else:
                ref.near = nearest_landmark(world, pid)
        truth = [oid for oid in present if satisfies(world, oid, now[oid], ref)]
        mem = {oid: p for oid, p in history.state_at(t0).items() if p != ABSENT}
        cases.append(GroundingCase(ref, target, truth, t0, dt, mem))
    return cases


class MapOnlyResolver:
    name = "Map-only"

    def __init__(self, world):
        self.world = world

    def scores(self, case):
        ids = sorted(case.memory)
        s = np.array([1.0 if satisfies(self.world, i, case.memory[i], case.ref) else 0.02 for i in ids])
        return ids, s / s.sum()


class StaleAwareResolver:
    name = "Staleness-aware"

    def __init__(self, world, prior, displacement):
        self.world, self.prior, self.disp = world, prior, displacement

    def scores(self, case):
        ids = sorted(case.memory)
        s = []
        P = len(self.world.places)
        for i in ids:
            cls = self.world.objects[i].cls
            if cls != case.ref.cls:
                s.append(1e-3)
                continue
            p = float(np.clip(self.prior.predict(cls, case.t0, case.dt), 1e-4, 1 - 1e-4))
            b = make_belief(self.world, self.disp, cls, case.memory[i], p, self.prior.absent_given_moved(cls, case.dt))
            sat = np.array([satisfies(self.world, i, pid, case.ref) for pid in range(P)], dtype=float)
            s.append(float(b[:P] @ sat) + 1e-3)
        s = np.array(s)
        return ids, s / s.sum()


def conformal_threshold(resolver, cal_cases, alpha):
    scores = []
    for c in cal_cases:
        if c.ambiguous or c.target not in c.memory:
            continue
        ids, p = resolver.scores(c)
        scores.append(1.0 - p[ids.index(c.target)])
    scores = np.sort(scores)
    n = len(scores)
    k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(scores[k])


def evaluate(resolver, cases, q):
    rows = []
    for c in cases:
        ids, p = resolver.scores(c)
        pred_set = [i for i, pi in zip(ids, p) if 1.0 - pi <= q]
        top = ids[int(np.argmax(p))]
        rows.append(dict(ambiguous=c.ambiguous, in_memory=c.target in c.memory,
                         ask=len(pred_set) != 1, act_id=pred_set[0] if len(pred_set) == 1 else None,
                         covered=c.target in pred_set, top=top, top_p=float(p.max()),
                         target=c.target, truth=c.truth))
    return rows


def summarise(rows):
    un = [r for r in rows if not r["ambiguous"] and r["in_memory"]]
    amb = [r for r in rows if r["ambiguous"]]
    acted = [r for r in un if not r["ask"]]
    return dict(
        n=len(rows), n_unambiguous=len(un), n_ambiguous=len(amb),
        argmax_accuracy=np.mean([r["top"] == r["target"] for r in un]) if un else np.nan,
        coverage=np.mean([r["covered"] for r in un]) if un else np.nan,
        ask_rate=np.mean([r["ask"] for r in rows]),
        ask_rate_ambiguous=np.mean([r["ask"] for r in amb]) if amb else np.nan,
        accuracy_when_acting=np.mean([r["act_id"] == r["target"] for r in acted]) if acted else np.nan,
        acted_fraction_unambiguous=len(acted) / len(un) if un else np.nan,
    )


def risk_coverage(rows):
    un = [r for r in rows if not r["ambiguous"] and r["in_memory"]]
    order = sorted(un, key=lambda r: -r["top_p"])
    err = np.cumsum([r["top"] != r["target"] for r in order])
    k = np.arange(1, len(order) + 1)
    return k / len(order), err / k
