"""3D line-of-sight with walls (full height) and boxes (finite height), plus a 2D occupancy grid for paths."""
import heapq
import math

import numpy as np


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, q1, q2):
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0:
        return True
    return False


def clip_segment_box(p, q, box):
    """Liang-Barsky: parameter interval [s0, s1] of segment p->q inside the box's 2D footprint, or None."""
    s0, s1 = 0.0, 1.0
    dx, dy = q[0] - p[0], q[1] - p[1]
    for pk, qk in ((-dx, p[0] - box.x0), (dx, box.x1 - p[0]), (-dy, p[1] - box.y0), (dy, box.y1 - p[1])):
        if pk == 0:
            if qk < 0:
                return None
        else:
            r = qk / pk
            if pk < 0:
                s0 = max(s0, r)
            else:
                s1 = min(s1, r)
            if s0 > s1:
                return None
    return s0, s1


def line_of_sight(world, eye, tgt):
    """eye, tgt: (x, y, z). Walls block at any height; a box blocks if the ray dips below its top inside it."""
    p, q = eye[:2], tgt[:2]
    for a, b in world.walls:
        if segments_intersect(p, q, a, b):
            return False
    for box in world.boxes:
        iv = clip_segment_box(p, q, box)
        if iv is None:
            continue
        s0, s1 = iv
        z0 = eye[2] + (tgt[2] - eye[2]) * s0
        z1 = eye[2] + (tgt[2] - eye[2]) * s1
        if min(z0, z1) < box.h - 1e-9:
            return False
    return True


def detection_prob(world, eye, tgt, max_range):
    d = math.dist(eye, tgt)
    if d > max_range or not line_of_sight(world, eye, tgt):
        return 0.0
    return 0.95 if d <= 3.0 else max(0.0, 0.95 - 0.15 * (d - 3.0))


def _point_seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


class Grid:
    """Occupancy grid for horizontal path lengths (8-connected Dijkstra)."""

    def __init__(self, world, res, inflation):
        self.res = res
        self.nx = int(round(world.width / res))
        self.ny = int(round(world.height / res))
        xs = (np.arange(self.nx) + 0.5) * res
        ys = (np.arange(self.ny) + 0.5) * res
        self.cx, self.cy = np.meshgrid(xs, ys, indexing="ij")
        blocked = np.zeros((self.nx, self.ny), dtype=bool)
        margin = 0.2
        blocked |= (self.cx < margin) | (self.cx > world.width - margin)
        blocked |= (self.cy < margin) | (self.cy > world.height - margin)
        for b in world.boxes:
            if b.blocks_path:
                blocked |= ((self.cx >= b.x0 - inflation) & (self.cx <= b.x1 + inflation)
                            & (self.cy >= b.y0 - inflation) & (self.cy <= b.y1 + inflation))
        for i in range(self.nx):
            for j in range(self.ny):
                if not blocked[i, j]:
                    for a, b in world.walls:
                        if _point_seg_dist(self.cx[i, j], self.cy[i, j], a, b) < 0.2:
                            blocked[i, j] = True
                            break
        self.blocked = blocked

    def cell_of(self, x, y):
        return (min(self.nx - 1, max(0, int(x / self.res))), min(self.ny - 1, max(0, int(y / self.res))))

    def center(self, cell):
        return float(self.cx[cell]), float(self.cy[cell])

    def free(self, cell):
        return not self.blocked[cell]

    def nearest_free(self, x, y):
        free = np.argwhere(~self.blocked)
        d = (self.cx[~self.blocked] - x) ** 2 + (self.cy[~self.blocked] - y) ** 2
        i = int(np.argmin(d))
        return tuple(int(v) for v in free[i])

    def dijkstra(self, start):
        dist = np.full((self.nx, self.ny), np.inf)
        dist[start] = 0.0
        pq = [(0.0, start)]
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        nx, ny, blocked, res = self.nx, self.ny, self.blocked, self.res
        while pq:
            d, (i, j) = heapq.heappop(pq)
            if d > dist[i, j]:
                continue
            for di, dj, w in steps:
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and not blocked[a, b]:
                    if di and dj and (blocked[i + di, j] or blocked[i, j + dj]):
                        continue  # no corner cutting
                    nd = d + w * res
                    if nd < dist[a, b]:
                        dist[a, b] = nd
                        heapq.heappush(pq, (nd, (a, b)))
        return dist
