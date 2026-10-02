"""Drone model: viewpoints (node x altitude), travel time/energy between them, and the visibility matrix.

A viewpoint (vp) is a free grid node at one of the configured altitudes. `V[vp, place]` is the probability
the drone detects an object resting on `place` when hovering at `vp` (line of sight x range falloff).
"""
import numpy as np

from .geometry import Grid, detection_prob
from .visibility import GROUP_OF, VisParams, group_footprint, visible_pixels, detect_prob


class DroneModel:
    def __init__(self, world, cfg):
        self.world, self.cfg = world, cfg
        grid = Grid(world, cfg.grid_res, cfg.path_inflation)
        self.grid = grid
        alts = tuple(sorted(set(cfg.altitudes) | {cfg.fixed_altitude, cfg.cruise_altitude}))
        self.alts = np.array(alts)

        # nodes: dock, vantage lattice, approach cells around every place
        nodes, index = [], {}

        def node_for(cell):
            if cell not in index:
                index[cell] = len(nodes)
                nodes.append(cell)
            return index[cell]

        dock_cell = grid.nearest_free(*world.dock)
        self.dock_node = node_for(dock_cell)
        reachable = np.isfinite(grid.dijkstra(dock_cell))  # pockets the drone cannot reach are excluded
        s = cfg.vantage_spacing
        for x in np.arange(s / 2, world.width, s):
            for y in np.arange(s / 2, world.height, s):
                c = grid.cell_of(x, y)
                if reachable[c]:
                    node_for(c)
        n_vantage_nodes = len(nodes)

        # candidate approach cells per place: reachable cells within the approach ring
        lo, hi = cfg.approach_ring
        free_idx = np.argwhere(reachable)
        fx, fy = grid.cx[reachable], grid.cy[reachable]
        self.approach_node = {}  # (pid, alt) -> node
        approach_cands = {}
        for p in world.places:
            d = np.hypot(fx - p.x, fy - p.y)
            ring = np.where((d >= lo) & (d <= hi))[0]
            if len(ring) == 0:
                ring = np.argsort(d)[:3]
            approach_cands[p.pid] = [(tuple(int(v) for v in free_idx[i]), d[i]) for i in ring]
        for p in world.places:
            tgt = (p.x, p.y, p.z + cfg.object_height)
            for a in alts:
                best = None
                for cell, dist in approach_cands[p.pid]:
                    cx, cy = grid.center(cell)
                    v = detection_prob(world, (cx, cy, a), tgt, cfg.max_range)
                    key = (v, -dist)
                    if best is None or key > best[0]:
                        best = (key, cell)
                self.approach_node[(p.pid, a)] = node_for(best[1])

        self.nodes = nodes
        self.node_xy = np.array([grid.center(c) for c in nodes])
        self.is_vantage = np.zeros(len(nodes), dtype=bool)
        self.is_vantage[:n_vantage_nodes] = True

        # horizontal path lengths between all nodes
        n = len(nodes)
        D = np.zeros((n, n))
        cells = np.array(nodes)
        for i, c in enumerate(nodes):
            dist = grid.dijkstra(c)
            D[i] = dist[cells[:, 0], cells[:, 1]]
        self.D = np.minimum(D, D.T)

        # viewpoints = node x altitude
        self.vp_node = np.repeat(np.arange(n), len(alts))
        self.vp_alt = np.tile(self.alts, n)
        self.n_vp = len(self.vp_node)

        # time and energy between viewpoints
        H = self.D[self.vp_node][:, self.vp_node]
        dz = self.vp_alt[None, :] - self.vp_alt[:, None]
        self.TIME = H / cfg.v_xy + np.abs(dz) / cfg.v_z
        climb_j = np.maximum(dz, 0.0) * cfg.mass_kg * 9.81 / cfg.climb_efficiency
        self.ENERGY = cfg.hover_power_w * self.TIME + climb_j
        self.COST = self.TIME + cfg.energy_weight * self.ENERGY
        obs_t = cfg.observe_time_s
        self.obs_time, self.obs_energy = obs_t, cfg.hover_power_w * obs_t
        self.obs_cost = obs_t + cfg.energy_weight * self.obs_energy

        # visibility matrix
        P = len(world.places)
        V = np.zeros((self.n_vp, P))
        tgts = [(p.x, p.y, p.z + cfg.object_height) for p in world.places]
        for v in range(self.n_vp):
            ex, ey = self.node_xy[self.vp_node[v]]
            eye = (ex, ey, float(self.vp_alt[v]))
            for j, t in enumerate(tgts):
                if abs(ex - t[0]) + abs(ey - t[1]) <= cfg.max_range * 1.5:
                    V[v, j] = detection_prob(world, eye, t, cfg.max_range)
        self.V = V
        self.V_group = {}
        if getattr(cfg, "visibility_model", "point") == "extended":
            self.V_group = self._extended_matrices(world, cfg)
            self.V = self.V_group["medium"]

    def _extended_matrices(self, world, cfg):
        """One visibility matrix per object size group (see visibility.py)."""
        prm = VisParams(a50_px=cfg.vis_a50_px, slope=cfg.vis_slope)
        out = {}
        for group in sorted(set(GROUP_OF.values())):
            width, height, top = group_footprint(group, getattr(cfg, "laptop_open", False))
            M = np.zeros((self.n_vp, len(world.places)))
            for v in range(self.n_vp):
                ex, ey = self.node_xy[self.vp_node[v]]
                eye = (ex, ey, float(self.vp_alt[v]))
                for j, p in enumerate(world.places):
                    M[v, j] = detect_prob(visible_pixels(world, eye, p.x, p.y, p.z, width, height, top, prm), prm)
            out[group] = M
        return out

    def V_for(self, cls):
        """Visibility matrix for objects of class `cls` (size-group specific under the extended model)."""
        return self.V_group.get(GROUP_OF.get(cls), self.V) if self.V_group else self.V

    # ---- helpers -----------------------------------------------------------------
    def vp(self, node, alt):
        return int(node * len(self.alts) + int(np.argmin(np.abs(self.alts - alt))))

    def dock_vp(self, alt):
        return self.vp(self.dock_node, alt)

    def approach_vp(self, pid, alt):
        a = float(self.alts[int(np.argmin(np.abs(self.alts - alt)))])
        return self.vp(self.approach_node[(pid, a)], a)

    def allowed_vps(self, altitudes):
        mask = np.isin(np.round(self.vp_alt, 3), np.round(np.array(altitudes), 3))
        return np.where(mask)[0]
