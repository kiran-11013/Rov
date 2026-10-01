"""Static world: three connected rooms, furniture/occluders, object resting places, object classes and instances.

Layout (metres), 18 x 8:
    Lab      x 0-10   desks, low cubicle partitions (1.3 m), a tall shelf, a side table
    Corridor x 10-12  door to lab at y 3.5-4.5
    Store    x 12-18  door from corridor at y 5.5-6.5, two tall shelving units (2.0 m), a table

Low partitions are the reason altitude matters: from 0.4 m they hide the desks behind them,
from 1.8 m the drone sees over them. Tall shelves hide one aisle from the next at any altitude.
"""
from dataclasses import dataclass, field

import numpy as np

COLOURS = ("red", "blue", "green", "black", "white", "grey", "yellow")


@dataclass(frozen=True)
class Box:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    h: float
    blocks_path: bool = True

    def contains(self, x, y, margin=0.0):
        return (self.x0 - margin <= x <= self.x1 + margin) and (self.y0 - margin <= y <= self.y1 + margin)


@dataclass(frozen=True)
class Place:
    pid: int
    room: str
    x: float
    y: float
    z: float
    kind: str  # 'floor' | 'table' | 'shelf'


@dataclass(frozen=True)
class ObjectClass:
    name: str
    kinds: tuple            # place kinds the class can rest on
    weibull_k: float        # shape of time-to-move, measured in *activity* hours
    weibull_scale_h: float  # scale in activity hours
    same_room: float        # probability a move stays in the same room
    periodic: bool = False  # leaves in the evening, returns next working morning
    identical: bool = False  # instances look the same (re-ID stress test)
    commonsense_median_h: float = 24.0  # "LLM-like" wall-clock median guess (stand-in prior)


# name, kinds, k, scale(active h), same_room, periodic, identical, commonsense median (wall h), count
CLASS_TABLE = [
    ("chair",   ("floor",),          0.8,    8.0, 0.90, False, True,     4.0, 12),
    ("stool",   ("floor",),          0.9,   30.0, 0.80, False, False,   12.0, 4),
    ("cart",    ("floor",),          1.0,   12.0, 0.40, False, False,    6.0, 3),
    ("bag",     ("floor", "table"),  1.0,   40.0, 0.80, True,  False,   10.0, 6),
    ("laptop",  ("table",),          1.0,   30.0, 0.80, True,  False,    8.0, 6),
    ("monitor", ("table",),          1.0, 1500.0, 0.90, False, False, 2000.0, 6),
    ("box",     ("floor", "shelf"),  0.9,   80.0, 0.50, False, False,   72.0, 8),
    ("bin",     ("floor",),          1.0,  300.0, 0.90, False, False,  500.0, 4),
    ("plant",   ("floor", "table"),  1.0, 2500.0, 0.90, False, False, 2000.0, 3),
    ("toolbox", ("table", "shelf"),  0.9,   25.0, 0.60, False, False,   24.0, 5),
    ("mug",     ("table",),          1.2,    5.0, 0.90, False, False,    3.0, 7),
]


@dataclass
class ObjectInstance:
    oid: int
    cls: str
    colour: str
    appearance: np.ndarray


@dataclass
class World:
    rooms: dict
    walls: list
    boxes: list
    places: list
    classes: dict
    objects: list
    landmarks: dict
    width: float = 18.0
    height: float = 8.0
    dock: tuple = (11.0, 3.0)
    _compat: dict = field(default_factory=dict, repr=False)

    def room_of(self, x, y):
        for name, (x0, y0, x1, y1) in self.rooms.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return None

    def compatible_places(self, cls_name):
        if cls_name not in self._compat:
            kinds = self.classes[cls_name].kinds
            self._compat[cls_name] = np.array([p.pid for p in self.places if p.kind in kinds], dtype=int)
        return self._compat[cls_name]

    def obj(self, oid):
        return self.objects[oid]


def _desk_places(box, room, start_pid):
    inset_x, inset_y = 0.4, 0.2
    pts = [(box.x0 + inset_x, box.y0 + inset_y), (box.x1 - inset_x, box.y0 + inset_y),
           (box.x0 + inset_x, box.y1 - inset_y), (box.x1 - inset_x, box.y1 - inset_y)]
    return [Place(start_pid + i, room, x, y, box.h, "table") for i, (x, y) in enumerate(pts)]


def make_world(seed=0, appearance_dim=16, layout="open"):
    """layout='open': short free-standing partitions (sight lines run around their ends).
    layout='cubicle': each lab desk is enclosed on three sides by 1.4 m cubicle walls, open to the aisle."""
    if layout not in ("open", "cubicle"):
        raise ValueError(f"unknown layout {layout!r}")
    rng = np.random.default_rng(seed)
    rooms = {"lab": (0.0, 0.0, 10.0, 8.0), "corridor": (10.0, 0.0, 12.0, 8.0), "store": (12.0, 0.0, 18.0, 8.0)}
    walls = [((10.0, 0.0), (10.0, 3.5)), ((10.0, 4.5), (10.0, 8.0)),
             ((12.0, 0.0), (12.0, 5.5)), ((12.0, 6.5), (12.0, 8.0))]

    desks = [Box("desk1", 1.2, 1.1, 2.8, 1.9, 0.75), Box("desk2", 1.2, 6.1, 2.8, 6.9, 0.75),
             Box("desk3", 5.2, 1.1, 6.8, 1.9, 0.75), Box("desk4", 5.2, 6.1, 6.8, 6.9, 0.75)]
    side_table = Box("side_table", 0.2, 3.0, 0.9, 5.0, 0.75)
    store_table = Box("store_table", 13.4, 6.6, 15.2, 7.6, 0.75)
    if layout == "open":
        partitions = [Box("part1", 3.9, 0.2, 4.1, 3.0, 1.3), Box("part2", 3.9, 5.0, 4.1, 7.8, 1.3),
                      Box("part3", 7.9, 0.2, 8.1, 3.0, 1.3), Box("part4", 7.9, 5.0, 8.1, 7.8, 1.3)]
    else:  # side walls for every desk, open toward the central aisle (y 3-5)
        partitions = []
        for i, (xa, xb) in enumerate(((0.85, 3.15), (4.85, 7.15))):
            for j, (ya, yb) in enumerate(((0.25, 2.9), (5.1, 7.75))):
                partitions.append(Box(f"cub{i}{j}w", xa, ya, xa + 0.1, yb, 1.4))
                partitions.append(Box(f"cub{i}{j}e", xb - 0.1, ya, xb, yb, 1.4))
    lab_shelf = Box("lab_shelf", 8.7, 0.3, 9.7, 1.0, 2.0)
    store_shelves = [Box("shelf_a", 13.0, 2.0, 17.0, 2.6, 2.0), Box("shelf_b", 13.0, 4.6, 17.0, 5.2, 2.0)]
    boxes = desks + [side_table, store_table] + partitions + [lab_shelf] + store_shelves

    places = []

    def add(room, x, y, z, kind):
        places.append(Place(len(places), room, x, y, z, kind))

    for d in desks:
        places.extend(_desk_places(d, "lab", len(places)))
    for y in (3.4, 4.0, 4.6):
        add("lab", 0.7, y, 0.75, "table")
    for x, y in ((13.8, 6.8), (14.8, 6.8), (14.3, 7.4)):
        add("store", x, y, 0.75, "table")

    floor_lab = [(1.6, 2.5), (2.4, 2.5), (1.6, 5.5), (2.4, 5.5), (5.6, 2.5), (6.4, 2.5), (5.6, 5.5), (6.4, 5.5),
                 (1.6, 0.6), (2.4, 0.6), (1.6, 7.4), (2.4, 7.4), (5.6, 0.6), (6.4, 0.6), (5.6, 7.4), (6.4, 7.4),
                 (3.0, 3.6), (3.0, 4.4), (5.0, 3.6), (5.0, 4.4), (7.0, 3.6), (7.0, 4.4), (9.2, 3.0), (9.2, 5.0),
                 (9.4, 7.4), (4.6, 7.4), (4.6, 0.6), (0.5, 7.4), (0.5, 0.6)]
    if layout == "cubicle":  # chairs live on the aisle side of an enclosed desk, not behind it
        behind_desk = {(1.6, 0.6): (1.25, 2.75), (2.4, 0.6): (2.75, 2.75), (5.6, 0.6): (5.25, 2.75),
                       (6.4, 0.6): (6.75, 2.75), (1.6, 7.4): (1.25, 5.25), (2.4, 7.4): (2.75, 5.25),
                       (5.6, 7.4): (5.25, 5.25), (6.4, 7.4): (6.75, 5.25),
                       (0.5, 0.6): (3.5, 0.6), (0.5, 7.4): (3.5, 7.4)}  # corners walled off by cubicles
        floor_lab = [behind_desk.get(pt, pt) for pt in floor_lab]
    for x, y in floor_lab:
        add("lab", x, y, 0.0, "floor")
    for x, y in ((11.0, 1.0), (11.0, 0.5), (11.0, 5.0), (11.0, 7.0), (11.0, 7.6), (10.6, 2.0)):
        add("corridor", x, y, 0.0, "floor")
    for x, y in ((13.0, 1.0), (15.0, 1.0), (17.0, 1.0), (13.0, 3.7), (15.0, 3.7), (17.0, 3.7), (17.3, 6.5), (12.6, 7.5)):
        add("store", x, y, 0.0, "floor")

    for x in (8.95, 9.45):
        for z in (0.4, 1.0, 1.6):
            add("lab", x, 1.15, z, "shelf")
    for sb in store_shelves:
        for face_y in (sb.y0 - 0.15, sb.y1 + 0.15):
            for x in (13.5, 14.5, 15.5, 16.5):
                for z in (0.4, 1.0, 1.6):
                    add("store", x, face_y, z, "shelf")

    classes = {}
    for name, kinds, k, scale, same, periodic, identical, cs, _n in CLASS_TABLE:
        classes[name] = ObjectClass(name, kinds, k, scale, same, periodic, identical, cs)

    # appearance embeddings: class base + colour + instance deviation (tiny for identical classes)
    base = {c: _unit(rng.normal(size=appearance_dim)) for c in classes}
    colour_vec = {c: _unit(rng.normal(size=appearance_dim)) for c in COLOURS}
    objects = []
    for name, *_rest, n in CLASS_TABLE:
        cl = classes[name]
        for _ in range(n):
            colour = "black" if cl.identical else COLOURS[rng.integers(len(COLOURS))]
            sigma = 0.02 if cl.identical else 0.5
            app = _unit(base[name] + 0.3 * colour_vec[colour] + sigma * rng.normal(size=appearance_dim))
            objects.append(ObjectInstance(len(objects), name, colour, app))

    landmarks = {"lab door": (10.0, 4.0), "store door": (12.0, 6.0), "whiteboard": (0.1, 4.0),
                 "window": (5.0, 7.9), "lab shelf": (9.2, 0.6), "store table": (14.3, 7.1)}
    w = World(rooms, walls, boxes, places, classes, objects, landmarks)
    w.layout = layout
    return w


def _unit(v):
    return v / np.linalg.norm(v)
