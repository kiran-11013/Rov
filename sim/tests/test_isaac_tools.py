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
