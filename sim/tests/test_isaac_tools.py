"""Tests for the Isaac Sim helper scripts that run without Isaac Sim (exporter + analysis)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "isaac"))

import analyze_visibility as AV  # noqa: E402
import export_scene as ES  # noqa: E402
from lbyf.config import SimConfig  # noqa: E402


def _spec():
    return ES.build_spec(SimConfig.quick(layout="booth"), n_nodes=4, yaws=(0, 180))


def test_export_spec_is_consistent_and_json_serialisable():
    spec = _spec()
    json.dumps(spec)
    oids = {o["oid"] for o in spec["objects"]}
    assert spec["objects"] and spec["viewpoints"]
    assert len(spec["viewpoints"]) == 4 * len(SimConfig().altitudes)
    for vp in spec["viewpoints"]:
        assert {int(k) for k in vp["model_p"]} == oids
        assert all(0.0 <= p <= 1.0 for p in vp["model_p"].values())
    assert all(o["cls"] in ES.CLASS_SHAPES for o in spec["objects"])
    assert spec["meta"]["weekday"] < 5 and 9 <= spec["meta"]["hour"] < 19


def test_analysis_perfect_and_flipped_render():
    spec = _spec()
    frames = []
    for vp in spec["viewpoints"]:
        for yaw in vp["yaws"]:
            px = {oid: (500 if p >= 0.5 else 0) for oid, p in vp["model_p"].items()} if yaw == 0 else {}
            frames.append({"node": vp["node"], "alt": vp["alt"], "yaw": yaw, "pixels": px})
    rows = AV.compare(spec, {"frames": frames})
    assert AV.summarise(rows)["agreement"] == 1.0
    _text, verdict, _s = AV.report(spec, rows, 200)
    assert verdict == "PASS"
    flipped = [{**f, "pixels": {oid: 500 for oid, p in vp["model_p"].items() if p < 0.5}}
               for f, vp in zip(frames[::2], spec["viewpoints"])]
    assert AV.summarise(AV.compare(spec, {"frames": flipped}))["agreement"] < 0.5


def test_kappa_bounds():
    m = np.array([1, 1, 0, 0], bool)
    assert AV.kappa(m, m) == 1.0
    assert AV.kappa(m, ~m) < 0


def test_calibration_recovers_known_threshold(tmp_path):
    """Fake renders generated from the full extended model with a50 = 400 px: calibration must pick '+top'
    and recover a threshold near 400."""
    import calibrate_visibility as CV
    from lbyf.visibility import VisParams, footprint, visible_pixels
    from lbyf.world import make_world

    paths = []
    for layout in ("booth", "cubicle"):
        cfg = SimConfig(layout=layout)
        spec = ES.build_spec(cfg, n_nodes=6)
        world = make_world(cfg.seed, **cfg.world_kwargs())
        prm = VisParams()
        frames = []
        for vp in spec["viewpoints"]:
            px = {}
            for ob in spec["objects"]:
                w, h, top = footprint(ob["shape"], ob["dims"])
                p = visible_pixels(world, (vp["x"], vp["y"], vp["alt"]), ob["x"], ob["y"], ob["z_base"], w, h, top, prm)
                if p > 0:
                    px[str(ob["oid"])] = int(p * 200 / 400)  # render >= 200 px  <=>  model >= 400 px
            frames.append({"node": vp["node"], "alt": vp["alt"], "yaw": 0, "pixels": px})
        path = tmp_path / f"{layout}.json"
        path.write_text(json.dumps({"frames": frames}))
        paths.append(f"{layout}={path}")

    orig = ES.build_spec
    CV.ES.build_spec = lambda cfg: orig(cfg, n_nodes=6)
    try:
        layouts = [CV.load_layout(*p.split("=", 1)) for p in paths]
        rows = [x for L in layouts for x in CV.pair_table(L, VisParams())]
        a50 = CV.fit_a50(rows)
        assert 300 <= a50 <= 520
        assert CV.score(rows, CV.threshold(rows, a50))["agreement"] > 0.97
    finally:
        CV.ES.build_spec = orig


def test_detection_matching_and_curve_fit(tmp_path):
    import json
    import math
    import sys

    import numpy as np
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "isaac"))
    import detect_frames as DF

    ids = np.zeros((100, 200), np.uint32)
    ids[10:30, 10:50] = 5    # obj 1 (mug)
    ids[50:90, 100:180] = 7  # obj 2 (chair)
    ids[60:70, 20:30] = 9    # not an object
    np.savez_compressed(tmp_path / "seg.npz", ids=ids, depth=np.zeros_like(ids, np.float16),
                        id_to_path=json.dumps({"5": "/World/Objects/obj_1/model/mesh", "7": "/World/Objects/obj_2",
                                               "9": "/World/Furniture/desk_0"}))
    gts = DF.gt_objects(tmp_path / "seg.npz")
    assert set(gts) == {1, 2} and gts[1][0] == 800 and gts[1][1] == (10, 10, 50, 30)
    cls_of = {1: "mug", 2: "chair"}
    dets = [("mug", 0.9, [11, 9, 49, 31]), ("mug", 0.4, [100, 50, 180, 90]), ("chair", 0.3, [95, 45, 185, 95])]
    assert DF.match(dets, gts, cls_of) == {1: 0.9, 2: 0.3}  # wrong-class box on the chair does not count

    rng = np.random.default_rng(0)
    px = np.exp(rng.uniform(math.log(5), math.log(20000), 4000))
    p = 0.95 / (1 + np.exp(-(np.log(px) - math.log(300)) / 0.4))
    fit = DF.fit_logistic(px, rng.random(4000) < p)
    assert 200 < fit["a50_px"] < 450 and fit["slope"] in (0.3, 0.4, 0.5)
