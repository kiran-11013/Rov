"""Belief over where a remembered object is now: mapped place, elsewhere (displacement prior), or absent.

The displacement prior is learned from training moves (FlowMaps-style, simplified): a Dirichlet-smoothed
room-transition table per class, uniform over compatible places inside the destination room.
"""
import numpy as np

from .dynamics import ABSENT


class DisplacementModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, history, world, t_end):
        self.world = world
        self.rooms = list(world.rooms)
        r_index = {r: i for i, r in enumerate(self.rooms)}
        self.counts = {c: np.zeros((len(self.rooms), len(self.rooms))) for c in world.classes}
        for oid, _t, p_from, p_to in history.moves(t_end):
            if p_from == ABSENT or p_to == ABSENT:
                continue
            c = world.objects[oid].cls
            self.counts[c][r_index[world.places[p_from].room], r_index[world.places[p_to].room]] += 1
        return self

    def place_distribution(self, cls, mapped_pid):
        """P(new place | moved), over all places (zero for incompatible and for the mapped place)."""
        world = self.world
        P = len(world.places)
        comp = world.compatible_places(cls)
        from_room = self.rooms.index(world.places[mapped_pid].room)
        row = self.counts[cls][from_room] + self.alpha
        rooms_avail = np.zeros(len(self.rooms), dtype=bool)
        per_room = {}
        for i, r in enumerate(self.rooms):
            pids = [pid for pid in comp if world.places[pid].room == r and pid != mapped_pid]
            per_room[i] = pids
            rooms_avail[i] = len(pids) > 0
        row = np.where(rooms_avail, row, 0.0)
        row = row / row.sum()
        dist = np.zeros(P)
        for i, pids in per_room.items():
            if pids:
                dist[pids] += row[i] / len(pids)
        return dist


def make_belief(world, displacement, cls, mapped_pid, p_still, q_absent):
    """Vector over places + one final entry for 'absent from the mapped space'."""
    P = len(world.places)
    b = np.zeros(P + 1)
    b[mapped_pid] = p_still
    b[P] = (1.0 - p_still) * q_absent
    b[:P] += (1.0 - p_still) * (1.0 - q_absent) * displacement.place_distribution(cls, mapped_pid)
    return b / b.sum()
