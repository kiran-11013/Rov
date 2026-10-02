"""Stage 2: stress-test the simulated claims across layouts, seeds, drone settings and look-alike objects.

Work is split into chunks (one configuration x a few seeds) and run in parallel. Every worker keeps its own
drone-matrix cache, so a layout is built once per worker. Workers return per-command vectors; all
statistics are computed afterwards in the parent from those vectors.

Statistics: seeds are the independent unit. Per-seed values are summarised by mean and a Student-t 95 %
interval across seeds.
"""
from multiprocessing import Pool

import numpy as np
from scipy import stats as sps

from . import experiments as X
from .config import SimConfig
from .episode import Arm
from .priors import labelled_pairs
from .stats import brier

_CACHE = {}


# ----------------------------------------------------------------------------- arm sets
def arms_altitude(ctx):
    o = ctx.ours
    return [Arm("trust", o, "trust"), Arm("trust@0.4", o, "trust", altitudes=(0.4,)),
            Arm("verify-always", o, "verify"), Arm("ours", o, "ours"),
            Arm("ours@0.4", o, "ours", altitudes=(0.4,)), Arm("ours@1.0", o, "ours", altitudes=(1.0,)),
            Arm("ours@1.8", o, "ours", altitudes=(1.8,)), Arm("oracle", None, "oracle")]


def arms_full(ctx):
    pf = ctx.priors["Persistence filter (exp.)"]
    cs = ctx.priors["Commonsense (LLM stand-in)"]
    return arms_altitude(ctx) + [Arm("ours+persistence-filter", pf, "ours"), Arm("ours+commonsense", cs, "ours")]


def arms_lookalike(ctx):
    o = ctx.ours
    return [Arm("trust", o, "trust"), Arm("trust (no spatial re-ID)", o, "trust", spatial_reid=False),
            Arm("verify-always", o, "verify"), Arm("ours", o, "ours"),
            Arm("ours (no spatial re-ID)", o, "ours", spatial_reid=False), Arm("oracle", None, "oracle")]


KINDS = {"altitude": arms_altitude, "full": arms_full, "lookalike": arms_lookalike}


# ----------------------------------------------------------------------------- worker
def _e1_point(ctx):
    rows = labelled_pairs(ctx.history, ctx.world, ctx.train_end, ctx.test_end, ctx.cfg.horizons_h)
    labels = np.array([r[4] for r in rows], float)
    dts = np.array([r[3] for r in rows])
    out = {}
    for name, prior in ctx.priors.items():
        b = brier(prior.predict_many(rows), labels)
        out[name] = {"all": float(b.mean()), **{f"{h:g}h": float(b[dts == h].mean()) for h in ctx.cfg.horizons_h}}
    return out


def run_chunk(task):
    recs = []
    for seed in task["seeds"]:
        cfg = SimConfig(seed=seed, **task["cfg"])
        ctx = X.setup(cfg, _CACHE)
        e2 = X.run_e2(ctx, KINDS[task["kind"]](ctx))
        cmds, world = e2["cmds"], ctx.world
        rec = {
            "tag": task["tag"], "group": task["group"], "seed": seed, "cfg": task["cfg"],
            "meta": {"moved": np.array([c.moved for c in cmds]), "horizon": np.array([c.dt for c in cmds]),
                     "oid": np.array([c.oid for c in cmds]),
                     "cls": np.array([world.objects[c.oid].cls for c in cmds]),
                     "ident": np.array([world.classes[world.objects[c.oid].cls].identical for c in cmds])},
            "arms": {name: {"succ": np.array([r.success for r in res], float),
                            "intent": np.array([r.success_intent for r in res], float),
                            "t": np.array([r.time_s for r in res]),
                            "en": np.array([r.energy_j for r in res]),
                            "wasted": np.array([r.wasted_looks for r in res], float)}
                     for name, res in e2["results"].items()},
        }
        if task.get("e1_e3"):
            rec["e1"] = _e1_point(ctx)
            rec["e3"] = {k: v["summary"] for k, v in X.run_e3(ctx).items()}
        recs.append(rec)
    return recs


def run_tasks(tasks, processes=4):
    with Pool(processes) as pool:
        out = []
        for i, recs in enumerate(pool.imap_unordered(run_chunk, tasks, chunksize=1), 1):
            out.extend(recs)
            print(f"  chunk {i}/{len(tasks)} done", flush=True)
    return out


def make_tasks(configs, kind, seeds, group, chunk=None, e1_e3=False):
    """configs: {tag: cfg_kwargs}. Seeds are split into chunks to use all cores."""
    chunk = chunk or len(seeds)
    tasks = []
    for tag, cfg in configs.items():
        for i in range(0, len(seeds), chunk):
            tasks.append({"tag": tag, "group": group, "cfg": cfg, "kind": kind, "seeds": list(seeds[i:i + chunk]),
                          "e1_e3": e1_e3})
    return tasks


# ----------------------------------------------------------------------------- statistics over seeds
def mean_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan
    h = float(sps.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h


def subset_mask(rec, subset):
    m = rec["meta"]
    if subset == "all":
        return np.ones(len(m["moved"]), bool)
    if subset == "moved":
        return m["moved"]
    if subset == "24h":
        return m["horizon"] == 24.0
    if subset == "1h":
        return m["horizon"] == 1.0
    if subset == "non-identical":
        return ~m["ident"]
    if subset == "identical":
        return m["ident"]
    raise ValueError(subset)


def arm_mean(rec, arm, key="t", subset="all"):
    m = subset_mask(rec, subset)
    return float(rec["arms"][arm][key][m].mean()) if m.any() else np.nan


def rel_gain(rec, base, new, key="t", subset="all"):
    """1 - mean(new)/mean(base): the fraction of `base` that `new` saves (time/energy)."""
    b, n = arm_mean(rec, base, key, subset), arm_mean(rec, new, key, subset)
    return 1.0 - n / b if b and np.isfinite(b) and np.isfinite(n) else np.nan


def diff(rec, a, b, key="succ", subset="all"):
    return arm_mean(rec, a, key, subset) - arm_mean(rec, b, key, subset)


def group_by_tag(records):
    out = {}
    for r in records:
        out.setdefault(r["tag"], []).append(r)
    return out
