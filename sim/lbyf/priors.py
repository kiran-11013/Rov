"""Persistence priors: P(object is still on its mapped place at t0 + dt | class, t0).

Compared in E1 and plugged into the action policy in E2:
* GlobalConstant          one number for everything (the "embarrassing" baseline)
* PersistenceFilterPrior  exponential survival per class in wall-clock time (Rosen et al.-style)
* CommonsensePrior        stand-in for an LLM / PreSIST-Lang prior: class-name commonsense medians, k = 1
* HumanGuessPrior         stand-in for 10 lab members' guesses: noisy commonsense, geometric mean
* HierarchicalWeibullPrior (ours) Weibull on *activity exposure* (time-of-day aware), per-class with
                          shrinkage toward the pooled fit, plus an empirical return component for
                          objects that leave and come back (bags, laptops).

The commonsense and human priors are stand-ins with documented, hand-set numbers. Real LLM / Jev /
human priors plug in through the same `predict` interface.
"""
import numpy as np

from .dynamics import ABSENT

RETURN_BUCKETS_H = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 24, 48, 72, 120, 168], dtype=float)


# ----------------------------------------------------------------------------- data
def residual_samples(history, world, t_start, t_end, step_h=2.0):
    """Residual time-on-place from sampled t0: rows (oid, cls, t0, tau, event)."""
    rows = []
    for oid in history.times:
        cls = world.objects[oid].cls
        for t0 in np.arange(t_start, t_end, step_h):
            if history.place_at(oid, t0) == ABSENT:
                continue
            tc = history.next_change_after(oid, t0)
            if tc < t_end:
                rows.append((oid, cls, t0, tc - t0, True))
            else:
                rows.append((oid, cls, t0, t_end - t0, False))
    return rows


def labelled_pairs(history, world, t_start, t_end, horizons, step_h=1.0, working_hours_only=True):
    """(oid, cls, t0, dt, still_there) for present objects; t0 restricted so t0+dt < t_end."""
    rows = []
    for oid in history.times:
        cls = world.objects[oid].cls
        for t0 in np.arange(t_start, t_end, step_h):
            if working_hours_only and not (_is_weekday(t0) and 9.0 <= t0 % 24 < 18.0):
                continue
            p0 = history.place_at(oid, t0)
            if p0 == ABSENT:
                continue
            for dt in horizons:
                if t0 + dt < t_end:
                    rows.append((oid, cls, t0, dt, history.place_at(oid, t0 + dt) == p0))
    return rows


def _is_weekday(t):
    return (int(t // 24) % 7) < 5


# ----------------------------------------------------------------------------- Weibull fitting
def fit_weibull_censored(e, event, k_grid=None):
    """Profile-likelihood MLE of a censored Weibull. Returns (k, scale, n_events)."""
    e = np.maximum(np.asarray(e, dtype=float), 1e-6)
    event = np.asarray(event, dtype=bool)
    d = int(event.sum())
    if d == 0:
        return 1.0, float(e.sum()) * 10.0 + 1.0, 0  # no events: very long scale
    if k_grid is None:
        k_grid = np.linspace(0.3, 3.0, 136)
    best = None
    log_e_ev = np.log(e[event]).sum()
    for k in k_grid:
        lam_k = (e ** k).sum() / d
        ll = d * np.log(k) - d * np.log(lam_k) + (k - 1) * log_e_ev - d
        if best is None or ll > best[0]:
            best = (ll, k, lam_k ** (1.0 / k))
    return float(best[1]), float(best[2]), d


# ----------------------------------------------------------------------------- priors
class Prior:
    name = "prior"

    def fit(self, history, world, t_end):
        return self

    def predict(self, cls, t0, dt):
        raise NotImplementedError

    def predict_many(self, rows):
        return np.array([self.predict(r[1], r[2], r[3]) for r in rows])


class GlobalConstant(Prior):
    name = "Global constant"

    def fit(self, history, world, t_end):
        rows = labelled_pairs(history, world, 0.0, t_end, (1.0, 24.0))
        self.p = float(np.mean([r[4] for r in rows])) if rows else 0.5
        return self

    def predict(self, cls, t0, dt):
        return self.p


class PersistenceFilterPrior(Prior):
    """Exponential survival per class in wall-clock time (no time of day, no returns)."""
    name = "Persistence filter (exp.)"

    def fit(self, history, world, t_end):
        rows = residual_samples(history, world, 0.0, t_end)
        self.rate = {}
        for c in world.classes:
            tau = np.array([r[3] for r in rows if r[1] == c])
            ev = np.array([r[4] for r in rows if r[1] == c])
            self.rate[c] = (ev.sum() + 0.5) / (tau.sum() + 1.0)
        return self

    def predict(self, cls, t0, dt):
        return float(np.exp(-self.rate[cls] * dt))


class CommonsensePrior(Prior):
    """Stand-in for an LLM/PreSIST-Lang prior: commonsense wall-clock medians, exponential shape."""
    name = "Commonsense (LLM stand-in)"

    def fit(self, history, world, t_end):
        self.median = {c: world.classes[c].commonsense_median_h for c in world.classes}
        return self

    def predict(self, cls, t0, dt):
        return float(2.0 ** (-dt / self.median[cls]))


class HumanGuessPrior(Prior):
    """Stand-in for 10 human guessers: commonsense median x lognormal(0, 0.6) each, geometric mean."""
    name = "Human guesses (stand-in)"

    def __init__(self, n_guessers=10, sigma=0.6, seed=0):
        self.n, self.sigma, self.seed = n_guessers, sigma, seed

    def fit(self, history, world, t_end):
        rng = np.random.default_rng(self.seed)
        self.median = {}
        for c in world.classes:
            g = world.classes[c].commonsense_median_h * np.exp(rng.normal(0, self.sigma, self.n))
            self.median[c] = float(np.exp(np.log(g).mean()))
        return self

    def predict(self, cls, t0, dt):
        return float(2.0 ** (-dt / self.median[cls]))


class HierarchicalWeibullPrior(Prior):
    """Ours: activity-exposure Weibull with class shrinkage + empirical return probability."""
    name = "Hier. Weibull + returns (ours)"

    def __init__(self, shrink_m=10.0):
        self.m = shrink_m

    def fit(self, history, world, t_end):
        self.act = history.activity
        rows = residual_samples(history, world, 0.0, t_end)
        exp_all = np.array([self.act.exposure(r[2] + r[3]) - self.act.exposure(r[2]) for r in rows])
        ev_all = np.array([r[4] for r in rows])
        cls_all = np.array([r[1] for r in rows])
        kp, sp, _ = fit_weibull_censored(exp_all, ev_all)
        self.params, self.events = {}, {}
        for c in world.classes:
            m = cls_all == c
            k, s, d = fit_weibull_censored(exp_all[m], ev_all[m])
            w = d / (d + self.m)
            self.params[c] = (np.exp(w * np.log(k) + (1 - w) * np.log(kp)),
                              np.exp(w * np.log(s) + (1 - w) * np.log(sp)))
            self.events[c] = d

        # empirical P(back on the mapped place | departed) and P(absent | departed) per class and horizon
        pairs = labelled_pairs(history, world, 0.0, t_end, tuple(RETURN_BUCKETS_H), step_h=2.0,
                               working_hours_only=False)
        nb = len(RETURN_BUCKETS_H)
        counts = {c: np.zeros((3, nb)) for c in world.classes}  # rows: departed, back, absent
        bucket_of = {float(h): i for i, h in enumerate(RETURN_BUCKETS_H)}
        for oid, cl, t0, h, _still in pairs:
            if history.next_change_after(oid, t0) >= t0 + h:
                continue  # never departed within h
            i = bucket_of[float(h)]
            p0, p1 = history.place_at(oid, t0), history.place_at(oid, t0 + h)
            counts[cl][0, i] += 1
            counts[cl][1, i] += p1 == p0
            counts[cl][2, i] += p1 == ABSENT
        self.q_return = {c: (v[1] + 0.1) / (v[0] + 1.0) for c, v in counts.items()}
        self.q_absent = {c: (v[2] + 0.1) / (v[0] + 1.0) for c, v in counts.items()}
        return self

    def survival(self, cls, t0, dt):
        k, s = self.params[cls]
        e = self.act.exposure(t0 + dt) - self.act.exposure(t0)
        return float(np.exp(-(e / s) ** k))

    def _bucket(self, dt):
        return int(np.argmin(np.abs(np.log(RETURN_BUCKETS_H) - np.log(max(dt, 1e-3)))))

    def predict(self, cls, t0, dt):
        S = self.survival(cls, t0, dt)
        return float(S + (1 - S) * self.q_return[cls][self._bucket(dt)])

    def absent_given_moved(self, cls, dt):
        return float(self.q_absent[cls][self._bucket(dt)])


class OraclePrior(Prior):
    """Knows the ground truth (used only to bound what a perfect prior could achieve)."""
    name = "Oracle prior"

    def __init__(self, history, memory_place):
        self.h, self.mem = history, memory_place

    def predict_for(self, oid, t0, dt):
        return 1.0 if self.h.place_at(oid, t0 + dt) == self.mem[oid] else 0.0


def all_baseline_priors(seed=0):
    return [GlobalConstant(), PersistenceFilterPrior(), CommonsensePrior(), HumanGuessPrior(seed=seed),
            HierarchicalWeibullPrior()]
