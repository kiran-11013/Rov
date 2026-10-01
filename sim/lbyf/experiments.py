"""E1 (priors), E2 (command trials with all arms), E3 (grounding) and the decision-region analysis."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .belief import DisplacementModel, make_belief
from .drone import DroneModel
from .dynamics import ABSENT, simulate_history
from .episode import Arm, Executor, sample_commands
from .grounding import (MapOnlyResolver, StaleAwareResolver, conformal_threshold, evaluate, risk_coverage,
                        sample_cases, summarise)
from .priors import (CommonsensePrior, GlobalConstant, HierarchicalWeibullPrior, HumanGuessPrior,
                     PersistenceFilterPrior, labelled_pairs)
from .stats import brier, cluster_bootstrap, holm, mcnemar_exact
from .world import make_world


@dataclass
class Context:
    cfg: object
    world: object
    history: object
    drone: object
    train_end: float
    test_end: float
    priors: dict
    ours: object
    disp: object
    rng: object


def setup(cfg):
    rng = np.random.default_rng(cfg.seed)
    world = make_world(cfg.seed, layout=cfg.layout)
    history = simulate_history(world, cfg, rng)
    drone = DroneModel(world, cfg)
    train_end, test_end = cfg.train_days * 24.0, cfg.days * 24.0
    priors = {}
    for p in (GlobalConstant(), PersistenceFilterPrior(), CommonsensePrior(), HumanGuessPrior(seed=cfg.seed),
              HierarchicalWeibullPrior()):
        priors[p.name] = p.fit(history, world, train_end)
    ours = priors[HierarchicalWeibullPrior.name]
    disp = DisplacementModel().fit(history, world, train_end)
    return Context(cfg, world, history, drone, train_end, test_end, priors, ours, disp, rng)


# ----------------------------------------------------------------------------- E1
def run_e1(ctx):
    cfg, world, h = ctx.cfg, ctx.world, ctx.history
    rows = labelled_pairs(h, world, ctx.train_end, ctx.test_end, cfg.horizons_h)
    labels = np.array([r[4] for r in rows], float)
    oids = np.array([r[0] for r in rows])
    dts = np.array([r[3] for r in rows])
    classes = np.array([r[1] for r in rows])
    rng = np.random.default_rng(cfg.seed + 1)
    table = []
    for name, prior in ctx.priors.items():
        pred = prior.predict_many(rows)
        b = brier(pred, labels)
        row = {"prior": name}
        for dt in cfg.horizons_h:
            m = dts == dt
            row[f"brier_{dt:g}h"] = cluster_bootstrap(b[m], oids[m], cfg.n_boot, rng)
        row["brier_all"] = cluster_bootstrap(b, oids, cfg.n_boot, rng)
        table.append(row)

    # per-class empirical persistence and training event counts
    train_moves = Counter(world.objects[o].cls for o, t, a, bb in h.moves(ctx.train_end))
    per_class = []
    for c in world.classes:
        r = {"class": c, "train_events": train_moves.get(c, 0)}
        for dt in cfg.horizons_h:
            m = (classes == c) & (dts == dt)
            r[f"emp_{dt:g}h"] = float(labels[m].mean()) if m.any() else np.nan
            r[f"ours_{dt:g}h"] = float(ctx.ours.predict_many([rows[i] for i in np.where(m)[0]]).mean()) if m.any() else np.nan
        per_class.append(r)
    return {"table": table, "per_class": per_class, "n_pairs": len(rows), "n_objects": len(set(oids))}


# ----------------------------------------------------------------------------- E2
def default_arms(ctx):
    pf = ctx.priors[PersistenceFilterPrior.name]
    cs = ctx.priors[CommonsensePrior.name]
    return [
        Arm("A1 Trust-then-search", ctx.ours, "trust"),
        Arm("A2 Verify-always", ctx.ours, "verify"),
        Arm("A3 Ours + persistence-filter prior", pf, "ours"),
        Arm("A4 Ours + commonsense prior (LLM stand-in)", cs, "ours"),
        Arm("A5 Ours", ctx.ours, "ours"),
        Arm("A1-fixed Trust-then-search @0.4 m", ctx.ours, "trust", fixed_height=True),
        Arm("A5-fixed Ours @0.4 m", ctx.ours, "ours", fixed_height=True),
        Arm("A6 Oracle", None, "oracle"),
    ]


def run_e2(ctx, arms=None):
    cfg = ctx.cfg
    arms = arms or default_arms(ctx)
    rng = np.random.default_rng(cfg.seed + 2)
    cmds = sample_commands(ctx.history, ctx.world, cfg, rng, ctx.train_end, ctx.test_end,
                           cfg.n_commands_per_horizon)
    ex = Executor(ctx.drone, ctx.world, cfg, ctx.disp, ctx.ours)
    results = {a.name: [] for a in arms}
    for cmd in cmds:
        state_now = ctx.history.state_at(cmd.t0 + cmd.dt)
        for a in arms:
            r_rng = np.random.default_rng((cfg.seed, cmd.idx))  # common random numbers across arms
            results[a.name].append(ex.run(a, cmd, state_now, r_rng))
    return {"cmds": cmds, "results": results, "arms": arms, "world": ctx.world}


def success_within_budget(e2, budgets):
    """P(success and time <= B): equals re-running with budget B, since trajectories are identical up to B."""
    out = {}
    for name, res in e2["results"].items():
        s = np.array([r.success for r in res])
        t = np.array([r.time_s for r in res])
        out[name] = np.array([np.mean(s & (t <= B)) for B in budgets])
    return out


def summarise_e2(e2, cfg):
    rng = np.random.default_rng(cfg.seed + 3)
    cmds = e2["cmds"]
    oid = np.array([c.oid for c in cmds])
    dts = np.array([c.dt for c in cmds])
    moved = np.array([c.moved for c in cmds])
    world = e2.get("world")
    non_ident = np.array([not world.classes[world.objects[c.oid].cls].identical for c in cmds]) if world else None
    rows = []
    for name, res in e2["results"].items():
        succ = np.array([r.success for r in res], float)
        intent = np.array([r.success_intent for r in res], float)
        t = np.array([r.time_s for r in res])
        en = np.array([r.energy_j for r in res])
        wasted = np.array([r.wasted_looks for r in res], float)
        acts = Counter(r.first_action for r in res)
        subsets = [("all", np.ones(len(cmds), bool))] + [(f"{dt:g}h", dts == dt) for dt in cfg.horizons_h] \
            + [("moved", moved)]
        if non_ident is not None:
            subsets.append(("non-identical classes", non_ident))
        for label, m in subsets:
            if not m.any():
                continue
            rows.append(dict(arm=name, subset=label, n=int(m.sum()),
                             success=cluster_bootstrap(succ[m], oid[m], cfg.n_boot, rng),
                             intent=float(intent[m].mean()),
                             time_mean=cluster_bootstrap(t[m], oid[m], cfg.n_boot, rng),
                             time_median=float(np.median(t[m])),
                             energy_kj=float(en[m].mean() / 1000.0),
                             wasted=float(wasted[m].mean()),
                             actions=dict(Counter(res[i].first_action for i in np.where(m)[0]))))
    return rows


def primary_test(e2):
    """Pre-registered: A5 vs the better of A1/A2 on instance success, paired, all commands."""
    R = e2["results"]
    s = {k: np.array([r.success for r in v]) for k, v in R.items()}
    a1, a2, a5 = s["A1 Trust-then-search"], s["A2 Verify-always"], s["A5 Ours"]
    best_name = "A1 Trust-then-search" if a1.mean() >= a2.mean() else "A2 Verify-always"
    p, n_ours_only, n_base_only = mcnemar_exact(a5, s[best_name])
    secondary = {}
    names = [("A5 vs A5-fixed (altitude)", "A5 Ours", "A5-fixed Ours @0.4 m"),
             ("A1 vs A1-fixed (altitude, trust)", "A1 Trust-then-search", "A1-fixed Trust-then-search @0.4 m"),
             ("A5 vs A3 (prior)", "A5 Ours", "A3 Ours + persistence-filter prior"),
             ("A5 vs A4 (prior)", "A5 Ours", "A4 Ours + commonsense prior (LLM stand-in)")]
    ps = []
    for label, x, y in names:
        pp, nx, ny = mcnemar_exact(s[x], s[y])
        ps.append(pp)
        secondary[label] = (pp, nx, ny)
    adj = holm(np.array(ps))
    for (label, _, _), a in zip(names, adj):
        secondary[label] = secondary[label] + (float(a),)
    # time comparison (paired) on commands both succeed
    t = {k: np.array([r.time_s for r in v]) for k, v in R.items()}
    both = a5 & s[best_name]
    dtime = float(np.mean(t["A5 Ours"][both] - t[best_name][both])) if both.any() else np.nan
    return dict(baseline=best_name, p=p, ours_only=n_ours_only, baseline_only=n_base_only,
                rate_ours=float(a5.mean()), rate_baseline=float(s[best_name].mean()),
                time_diff_s_when_both_succeed=dtime, secondary=secondary)


# ----------------------------------------------------------------------------- decision regions
def decision_regions(ctx, cls="laptop", mapped_pid=4, dt=24.0, n=41):
    """Expected cost of each first action vs P(still there) for one representative command."""
    cfg = ctx.cfg
    ex = Executor(ctx.drone, ctx.world, cfg, ctx.disp, ctx.ours)
    q_abs = ctx.ours.absent_given_moved(cls, dt)
    ps = np.linspace(0.0, 1.0, n)
    out = {}
    for fixed in (False, True):
        pl = ex.planners[fixed]
        cur = ctx.drone.dock_vp(cfg.fixed_altitude if fixed else cfg.cruise_altitude)
        costs = {"trust": [], "verify": [], "search": []}
        chosen = []
        for p in ps:
            b = make_belief(ctx.world, ctx.disp, cls, mapped_pid, float(np.clip(p, 1e-4, 1 - 1e-4)), q_abs)
            costs["trust"].append(pl.expected_cost_first(b, pl.trust_vp(mapped_pid), cur))
            v = pl.verify_candidates(mapped_pid, cur)
            costs["verify"].append(min(pl.expected_cost_first(b, x, cur) for x in v) if v else np.nan)
            s = pl.next_vp(b[:pl.P], cur)
            costs["search"].append(pl.expected_cost_first(b, s, cur))
            chosen.append(pl.decide(b, mapped_pid, cur)[0])
        out["fixed" if fixed else "full"] = {"p": ps, "costs": {k: np.array(v) for k, v in costs.items()},
                                             "chosen": chosen}
    return out


# ----------------------------------------------------------------------------- E3
def run_e3(ctx):
    cfg = ctx.cfg
    rng = np.random.default_rng(cfg.seed + 4)
    cases = sample_cases(ctx.history, ctx.world, rng, cfg.n_grounding, ctx.train_end, ctx.test_end,
                         cfg.horizons_h)
    half = len(cases) // 2
    cal, test = cases[:half], cases[half:]
    out = {}
    for res in (MapOnlyResolver(ctx.world), StaleAwareResolver(ctx.world, ctx.ours, ctx.disp)):
        q = conformal_threshold(res, cal, cfg.conformal_alpha)
        rows = evaluate(res, test, q)
        cov, risk = risk_coverage(rows)
        out[res.name] = {"q": q, "summary": summarise(rows), "risk_coverage": (cov, risk)}
    return out
