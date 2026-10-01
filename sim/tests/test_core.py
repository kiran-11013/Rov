import numpy as np
import pytest

from lbyf.config import SimConfig
from lbyf.dynamics import ABSENT, Activity, simulate_history
from lbyf.geometry import line_of_sight
from lbyf.priors import fit_weibull_censored
from lbyf.stats import holm, mcnemar_exact
from lbyf.world import make_world


@pytest.fixture(scope="module")
def ctx():
    from lbyf import experiments as X
    return X.setup(SimConfig.quick(n_commands_per_horizon=8))


def test_wall_blocks_line_of_sight():
    w = make_world(0)
    assert not line_of_sight(w, (8.0, 1.0, 1.0), (14.0, 1.0, 1.0))  # through the solid part of x=10 wall
    assert line_of_sight(w, (9.0, 4.0, 1.0), (11.0, 4.0, 1.0))      # through the lab door


def test_low_partition_blocks_low_but_not_high_viewpoint():
    w = make_world(0)
    target = (6.0, 1.5, 0.95)  # object on desk3
    assert not line_of_sight(w, (3.0, 1.5, 0.4), target)
    assert line_of_sight(w, (3.0, 1.5, 1.8), target)


def test_weibull_mle_recovers_parameters():
    rng = np.random.default_rng(1)
    t = 10.0 * rng.weibull(1.5, 4000)
    c = 15.0
    k, s, d = fit_weibull_censored(np.minimum(t, c), t < c)
    assert abs(k - 1.5) < 0.1 and abs(s - 10.0) < 0.5 and d == int((t < c).sum())


def test_activity_inverse():
    a = Activity(14)
    for t in (3.0, 10.5, 50.0, 130.0):
        assert abs(float(a.wall_time(a.exposure(t))) - t) < 0.02


def test_history_is_consistent():
    w, cfg = make_world(0), SimConfig.quick()
    h = simulate_history(w, cfg, np.random.default_rng(0))
    for t in (10.0, 100.0, 300.0):
        occupied = [p for p in h.state_at(t).values() if p != ABSENT]
        assert len(occupied) == len(set(occupied))  # one object per place
    periodic = [o.oid for o in w.objects if w.classes[o.cls].periodic]
    night = np.mean([h.place_at(o, 3 * 24 + 3.0) == ABSENT for o in periodic])  # Thursday 03:00
    assert night > 0.4


def test_mcnemar_and_holm():
    p, a, b = mcnemar_exact(np.zeros(6, bool), np.ones(6, bool))
    assert (a, b) == (0, 6) and abs(p - 2 / 64) < 1e-12
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_policy_expected_cost_decreases_with_confidence(ctx):
    from lbyf.belief import make_belief
    from lbyf.episode import Executor
    ex = Executor(ctx.drone, ctx.world, ctx.cfg, ctx.disp, ctx.ours)
    pl = ex.planners[False]
    cur = ctx.drone.dock_vp(ctx.cfg.cruise_altitude)
    costs = []
    for p in (0.1, 0.5, 0.99):
        b = make_belief(ctx.world, ctx.disp, "laptop", 4, p, 0.3)
        costs.append(pl.decide(b, 4, cur)[2])
    assert costs[0] > costs[1] > costs[2]


def test_every_place_is_observable(ctx):
    assert (ctx.drone.V.max(axis=0) > 0).all()
    assert np.isfinite(ctx.drone.D).all()


def test_e2_runs_and_oracle_succeeds(ctx):
    from lbyf import experiments as X
    e2 = X.run_e2(ctx)
    assert len(e2["cmds"]) > 0
    assert all(r.success for r in e2["results"]["A6 Oracle"])
    for res in e2["results"].values():
        assert all(r.time_s <= ctx.cfg.time_budget_s + 1e-9 for r in res)


def test_cubicle_layout_is_fully_observable():
    from lbyf.drone import DroneModel
    cfg = SimConfig.quick(layout="cubicle")
    d = DroneModel(make_world(0, layout="cubicle"), cfg)
    assert (d.V.max(axis=0) > 0).all() and np.isfinite(d.D).all()


def test_conformal_threshold_covers():
    from lbyf.grounding import conformal_threshold

    class FakeCase:
        def __init__(self, target, p):
            self.target, self.memory, self.ambiguous, self.p = target, {0: 0, 1: 0, 2: 0}, False, p

    class FakeResolver:
        def scores(self, case):
            return [0, 1, 2], case.p

    rng = np.random.default_rng(0)
    cases = []
    for _ in range(400):
        p = rng.dirichlet([2, 1, 1])
        cases.append(FakeCase(int(rng.choice(3, p=p)), p))
    q = conformal_threshold(FakeResolver(), cases[:200], 0.1)
    cov = np.mean([1 - c.p[c.target] <= q for c in cases[200:]])
    assert cov >= 0.85
