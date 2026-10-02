"""Trust / verify / search as a choice of the *first viewpoint*, followed by belief-greedy search.

Every action is "fly to viewpoint X, look, then keep searching greedily if the object was not found":
* trust   X = approach point of the mapped place (fly there and look)
* verify  X = a viewpoint that sees the mapped place cheaply, often from altitude
* search  X = the greedy choice under the full belief (ignores that the mapped place is special)

Greedy search picks the viewpoint maximising (belief mass it would see) / (travel + observe cost), the
classic probability-per-cost rule for search. Expected costs are computed by rolling that greedy rule
forward on the belief (no sampling), with a penalty for not finding the object within the budget.

Cost = time + lambda * energy (cfg.energy_weight), in seconds.
"""
import numpy as np


class Planner:
    def __init__(self, drone, altitudes, approach_alt, max_rollout_steps=15, V=None):
        self.d = drone
        self.V = drone.V if V is None else V  # visibility matrix for the target's size group
        self.allowed = drone.allowed_vps(altitudes)
        self.approach_alt = approach_alt
        P = self.V.shape[1]
        self.P = P
        self.app = np.array([drone.approach_vp(pid, approach_alt) for pid in range(P)])
        self.max_steps = max_rollout_steps
        cfg = drone.cfg
        self.fail_penalty = cfg.time_budget_s * (1 + cfg.energy_weight * cfg.hover_power_w)
        self.budget_cost = self.fail_penalty

    # --- greedy rule ----------------------------------------------------------------
    def next_vp(self, m, cur, exclude=None):
        """exclude: viewpoints already looked from in this episode (a repeat look adds nothing)."""
        gains = self.V[self.allowed] @ m
        cost = self.d.COST[cur, self.allowed] + self.d.obs_cost
        ratio = np.where(gains > 1e-9, gains / cost, -np.inf)
        if exclude:
            ratio[np.isin(self.allowed, list(exclude))] = -np.inf
        i = int(np.argmax(ratio))
        if not np.isfinite(ratio[i]):
            return None
        return int(self.allowed[i])

    # --- model-based expected cost --------------------------------------------------
    def _found_cost(self, found, vp):
        return float(found @ self.d.COST[vp, self.app])

    def expected_cost_from(self, m, absent, cur):
        """Expected remaining cost of greedy search from `cur`; m is the unnormalised joint
        P(object at place and not yet found)."""
        m = m.copy()
        E, spent = 0.0, 0.0
        for _ in range(self.max_steps):
            if m.sum() < 1e-6:
                break
            vp = self.next_vp(m, cur)
            if vp is None:
                break
            c = self.d.COST[cur, vp] + self.d.obs_cost
            E += (m.sum() + absent) * c
            spent += c
            found = m * self.V[vp]
            E += self._found_cost(found, vp)
            m -= found
            cur = vp
            if spent > self.budget_cost:
                break
        return E + (m.sum() + absent) * self.fail_penalty

    def expected_cost_first(self, b, first_vp, cur):
        m, absent = b[:self.P].copy(), float(b[self.P])
        c = self.d.COST[cur, first_vp] + self.d.obs_cost
        found = m * self.V[first_vp]
        rest = self.expected_cost_from(m - found, absent, first_vp)
        return c + self._found_cost(found, first_vp) + rest

    # --- the three first actions ----------------------------------------------------
    def trust_vp(self, mapped):
        return int(self.app[mapped])

    def verify_candidates(self, mapped, cur, k=6, min_vis=0.3):
        vis = self.V[self.allowed, mapped]
        ok = vis >= min_vis
        if not ok.any():
            return []
        ratio = np.where(ok, vis / (self.d.COST[cur, self.allowed] + self.d.obs_cost), -np.inf)
        order = np.argsort(-ratio)[:k]
        return [int(self.allowed[i]) for i in order if np.isfinite(ratio[i])]

    def best_verify_vp(self, mapped, cur):
        c = self.verify_candidates(mapped, cur, k=1)
        return c[0] if c else self.trust_vp(mapped)

    def decide(self, b, mapped, cur):
        """Ours: pick the first viewpoint with the lowest expected cost among trust / verify / search."""
        options = [("trust", self.trust_vp(mapped))]
        options += [("verify", vp) for vp in self.verify_candidates(mapped, cur)]
        s = self.next_vp(b[:self.P], cur)
        if s is not None:
            options.append(("search", s))
        scored = [(self.expected_cost_first(b, vp, cur), name, vp) for name, vp in options]
        E, name, vp = min(scored, key=lambda x: x[0])
        if name == "verify" and vp == self.trust_vp(mapped):
            name = "trust"
        return name, vp, E
