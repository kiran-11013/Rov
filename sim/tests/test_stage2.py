import numpy as np
import pytest

from lbyf import stage2 as S
from lbyf.config import SimConfig
from lbyf.drone import DroneModel
from lbyf.geometry import line_of_sight
from lbyf.world import make_world


def test_booth_wall_blocks_aisle_view_at_low_altitude_but_not_high():
    w = make_world(0, layout="booth", partition_h=1.4)
    target = (1.6, 1.5, 0.95)            # object on desk1, inside the booth
    low, high = (1.5, 3.2, 0.4), (1.5, 3.2, 1.8)  # just outside the front wall; a 1.8 m eye clears the 1.4 m wall
    assert not line_of_sight(w, low, target)
    assert line_of_sight(w, high, target)


def test_wall_taller_than_every_altitude_blocks_all_views():
    w = make_world(0, layout="booth", partition_h=2.0)
    assert not line_of_sight(w, (1.5, 4.0, 1.8), (1.6, 1.5, 0.95))


def test_clutter_hides_far_side_of_desk_from_low_viewpoint_only():
    w = make_world(0, clutter_h=0.3)
    near, far = (2.0, 1.15, 0.95), (2.0, 1.85, 0.95)   # either side of the clutter block on desk1 (y 1.1-1.9)
    eye_low, eye_high = (2.0, 0.6, 0.9), (2.0, 1.0, 1.8)
    assert line_of_sight(w, eye_low, near)
    assert not line_of_sight(w, eye_low, far)
    assert line_of_sight(w, eye_high, far)


@pytest.mark.parametrize("kw", [dict(layout="booth", partition_h=1.6, clutter_h=0.45), dict(layout="cubicle")])
def test_variants_are_reachable_and_observable(kw):
    d = DroneModel(make_world(0, **kw), SimConfig.quick(**kw))
    assert np.isfinite(d.D).all() and (d.V.max(axis=0) > 0).all()


def test_identical_class_controls():
    w = make_world(0, n_chairs=20, identical_classes=("chair", "mug"))
    assert sum(o.cls == "chair" for o in w.objects) == 20
    assert w.classes["mug"].identical and w.classes["chair"].identical and not w.classes["bag"].identical


def test_drone_cache_reuse_matches_fresh_build():
    from lbyf import experiments as X
    cache = {}
    a = X.setup(SimConfig.quick(seed=1), cache)
    b = X.setup(SimConfig.quick(seed=2, obs_noise=0.2), cache)  # same geometry key, different perception
    assert len(cache) == 1
    assert b.drone.cfg.obs_noise == 0.2 and a.drone.cfg.obs_noise == 0.05
    assert np.array_equal(a.drone.V, b.drone.V)


def test_altitude_arm_restricts_viewpoints():
    from lbyf import experiments as X
    ctx = X.setup(SimConfig.quick(n_commands_per_horizon=5))
    e2 = X.run_e2(ctx, S.arms_altitude(ctx))
    assert set(e2["results"]) >= {"ours", "ours@0.4", "trust", "oracle"}
    ex_alts = {a.name: a.altitudes for a in e2["arms"]}
    assert ex_alts["ours@1.8"] == (1.8,)


def test_mean_ci_and_gain():
    m, lo, hi = S.mean_ci([1.0, 2.0, 3.0, 4.0])
    assert abs(m - 2.5) < 1e-12 and lo < m < hi
    rec = {"arms": {"a": {"t": np.array([10.0, 10.0])}, "b": {"t": np.array([8.0, 8.0])}},
           "meta": {"moved": np.array([True, False]), "horizon": np.array([1.0, 24.0]),
                    "ident": np.array([False, False])}}
    assert abs(S.rel_gain(rec, "a", "b") - 0.2) < 1e-12


def test_seeing_over_an_occluder_needs_a_steep_angle():
    """Why altitude rarely helps: the same 1.4 m wall that a 1.8 m eye clears from 0.3 m away
    blocks it from 1.1 m away, because the sight line to a 0.95 m target grazes the wall top."""
    w = make_world(0, layout="booth", partition_h=1.4)
    target = (1.6, 1.5, 0.95)
    assert line_of_sight(w, (1.5, 3.2, 1.8), target)
    assert not line_of_sight(w, (1.5, 4.0, 1.8), target)


def test_replay_path_matches_simulated_time():
    import numpy as np
    from lbyf.config import SimConfig
    from lbyf.episode import Arm, Executor, sample_commands
    from lbyf.experiments import setup
    from lbyf.replay import flight_path, pose_at

    cfg = SimConfig.quick()
    ctx = setup(cfg)
    ex = Executor(ctx.drone, ctx.world, cfg, ctx.disp, ctx.ours)
    cmds = sample_commands(ctx.history, ctx.world, cfg, np.random.default_rng(1), ctx.train_end, ctx.test_end, 5)
    for cmd in cmds[:6]:
        for arm in (Arm("trust", ctx.ours, "trust"), Arm("ours", ctx.ours, "ours")):
            trace = []
            r = ex.run(arm, cmd, ctx.history.state_at(cmd.t0 + cmd.dt), np.random.default_rng(0), trace=trace)
            keys, events = flight_path(ctx.drone, trace)
            if r.success:
                assert abs(keys[-1][0] - r.time_s) < 1e-6
            assert all(k1[0] >= k0[0] for k0, k1 in zip(keys, keys[1:]))
            for k0, k1 in zip(keys, keys[1:]):  # never faster than the configured speeds
                dt = k1[0] - k0[0]
                assert np.hypot(k1[1] - k0[1], k1[2] - k0[2]) <= cfg.v_xy * dt + 1e-6
                assert abs(k1[3] - k0[3]) <= cfg.v_z * dt + 1e-6
            for k in keys:  # every waypoint is in free space (the drone never cuts through furniture)
                assert ctx.drone.grid.free(ctx.drone.grid.cell_of(k[1], k[2]))
            assert len(pose_at(keys, keys[-1][0] / 2)) == 4
