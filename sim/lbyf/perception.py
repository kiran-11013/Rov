"""Detection and re-identification at a viewpoint.

Detection uses the same visibility matrix the planner uses (model = reality for detection rates).
Re-identification compares a noisy observed appearance with the remembered one; identical-looking
instances (all chairs) exceed the threshold too, which is where instance-level mistakes come from.
"""
from dataclasses import dataclass

import numpy as np

from .dynamics import ABSENT


@dataclass
class Detection:
    oid: int        # ground-truth identity (hidden from the policy except through appearance)
    cls: str
    pid: int
    appearance: np.ndarray


def observe(drone, world, state, vp, rng, cls_filter=None):
    dets = []
    for oid, pid in state.items():
        if pid == ABSENT:
            continue
        o = world.objects[oid]
        if cls_filter is not None and o.cls != cls_filter:
            continue
        if rng.random() < drone.V_for(o.cls)[vp, pid]:
            a = o.appearance + rng.normal(0, drone.cfg.obs_noise, size=o.appearance.shape)
            dets.append(Detection(oid, o.cls, pid, a / np.linalg.norm(a)))
    return dets


def reidentify(detections, memory_appearance, threshold, belief=None, sharpness=50.0):
    """Accept the detection with the highest log-posterior among those above the appearance threshold.

    Score = sharpness * cosine similarity + log belief(place). Appearance dominates for distinct objects
    (similarities differ by ~0.5); for identical-looking objects (similarities within ~0.01) the spatial
    prior decides, so the chair on the remembered spot is preferred over its twin elsewhere.
    """
    best, best_score = None, -np.inf
    for d in detections:
        sim = float(d.appearance @ memory_appearance)
        if sim < threshold:
            continue
        score = sharpness * sim + (np.log(belief[d.pid] + 1e-6) if belief is not None else 0.0)
        if score > best_score:
            best, best_score = d, score
    return best
