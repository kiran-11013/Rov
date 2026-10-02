"""Extended visibility model, calibrated against Isaac Sim renders (Stage 3a).

The Stage 1-2 'point' model checks one point 0.2 m above an object's support. Renders showed it under-counts tall
objects peeking over occluders, over-counts small/flat ones, and therefore understates what altitude reveals.
The extended model treats an object as a box/cylinder of real size and estimates how many pixels of it a ZED-like
camera would see:

    side term  width x height x (fraction of sample points on the vertical axis that are in sight and in the
               camera's vertical field of view) x cos(elevation)
    top term   top area x sin(looking-down angle), if the top centre is in sight and below the eye
    pixels     focal_px^2 / distance^2 x (side + top)
    p(detect)  p_max * logistic((ln pixels - ln a50) / slope)

Horizontal field of view is not limited: every viewpoint is rendered at 4 yaws that together cover 360 degrees.
"""
import math
from dataclasses import dataclass

from .geometry import line_of_sight

# Proxy geometry per class (also used to build the Isaac Sim scene): ("box", (sx, sy, sz)) or ("cyl", (r, h)).
CLASS_DIMS = {
    "chair": ("box", (0.45, 0.45, 0.90)),
    "stool": ("cyl", (0.18, 0.45)),
    "cart": ("box", (0.80, 0.50, 0.90)),
    "bag": ("box", (0.35, 0.15, 0.40)),
    "laptop": ("box", (0.33, 0.23, 0.03)),
    "monitor": ("box", (0.55, 0.06, 0.35)),
    "box": ("box", (0.40, 0.30, 0.30)),
    "bin": ("cyl", (0.15, 0.40)),
    "plant": ("cyl", (0.12, 0.50)),
    "toolbox": ("box", (0.40, 0.20, 0.20)),
    "mug": ("cyl", (0.045, 0.10)),
}

OPEN_LAPTOP = ("box", (0.33, 0.23, 0.25))  # screen up: the tall part a low camera can see


def class_dims(cls, laptop_open=False):
    return OPEN_LAPTOP if (cls == "laptop" and laptop_open) else CLASS_DIMS[cls]


# Classes share one visibility matrix per size group (keeps the drone model fast).
GROUP_OF = {"chair": "tall", "cart": "tall", "bag": "medium", "box": "medium", "bin": "medium", "plant": "medium",
            "toolbox": "medium", "monitor": "medium", "stool": "medium", "laptop": "flat", "mug": "small"}


def footprint(shape, dims):
    """(apparent width, height, top area) of a proxy shape; width averages the two horizontal sides."""
    if shape == "box":
        sx, sy, sz = dims
        return (sx + sy) / 2, sz, sx * sy
    r, h = dims
    return 2 * r, h, math.pi * r * r


def class_footprint(cls, laptop_open=False):
    return footprint(*class_dims(cls, laptop_open))


def group_footprint(group, laptop_open=False):
    members = [class_footprint(c, laptop_open) for c, g in GROUP_OF.items() if g == group]
    return tuple(sum(m[i] for m in members) / len(members) for i in range(3))


@dataclass(frozen=True)
class VisParams:
    a50_px: float = 200.0        # pixel area at which detection probability is p_max / 2
    slope: float = 0.35          # logistic width in ln(pixels)
    p_max: float = 0.95
    hfov_deg: float = 110.0
    width_px: int = 1280
    height_px: int = 720
    pitch_down_deg: float = 15.0
    max_range: float = 15.0
    side_samples: tuple = (0.15, 0.5, 0.85)
    use_top: bool = True
    use_vfov: bool = True

    @property
    def focal_px(self):
        return (self.width_px / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def vfov_half_deg(self):
        return math.degrees(math.atan(math.tan(math.radians(self.hfov_deg) / 2) * self.height_px / self.width_px))


def _in_vfov(eye, pt, prm):
    if not prm.use_vfov:
        return True
    el = math.degrees(math.atan2(pt[2] - eye[2], math.hypot(pt[0] - eye[0], pt[1] - eye[1])))
    centre = -prm.pitch_down_deg
    return centre - prm.vfov_half_deg <= el <= centre + prm.vfov_half_deg


def visible_pixels(world, eye, x, y, z_base, width, height, top_area, prm):
    """Estimated visible pixel area of an object resting at (x, y, z_base)."""
    cz = z_base + height / 2
    d = math.dist(eye, (x, y, cz))
    if d > prm.max_range or d < 1e-6:
        return 0.0
    seen = 0
    for f in prm.side_samples:
        pt = (x, y, z_base + f * height)
        if _in_vfov(eye, pt, prm) and line_of_sight(world, eye, pt):
            seen += 1
    frac = seen / len(prm.side_samples)
    el = math.atan2(cz - eye[2], math.hypot(x - eye[0], y - eye[1]))
    area = width * height * frac * math.cos(el)
    if prm.use_top and eye[2] > z_base + height:
        top = (x, y, z_base + height + 1e-3)
        if _in_vfov(eye, top, prm) and line_of_sight(world, eye, top):
            area += top_area * math.sin(-el) if el < 0 else 0.0
    return prm.focal_px ** 2 * area / (d * d)


def detect_prob(px, prm):
    if px <= 0:
        return 0.0
    return prm.p_max / (1.0 + math.exp(-(math.log(px) - math.log(prm.a50_px)) / prm.slope))


def learned_detect_prob(px, eye, centre, coef):
    """Real-detector model fitted on Isaac Sim renders (isaac/fit_detection_model.py):
    sigmoid(b0 + b1 ln px + b2 down/30deg), down = how steeply the camera looks down at the object centre."""
    if px <= 0:
        return 0.0
    b0, b1, b2 = coef
    down = math.degrees(math.atan2(eye[2] - centre[2], max(math.hypot(centre[0] - eye[0], centre[1] - eye[1]), 1e-3)))
    z = b0 + b1 * math.log(px) + b2 * max(down, 0.0) / 30.0
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def detection_prob_extended(world, eye, x, y, z_base, width, height, top_area, prm):
    return detect_prob(visible_pixels(world, eye, x, y, z_base, width, height, top_area, prm), prm)
