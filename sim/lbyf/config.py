"""Simulation configuration. All times are in hours unless the name ends in `_s` (seconds)."""
from dataclasses import dataclass


@dataclass
class SimConfig:
    seed: int = 0
    layout: str = "open"       # 'open' | 'cubicle' (see world.make_world)

    # Persistence logging (E1)
    days: int = 42              # six weeks of simulated object movement
    train_days: int = 28        # first four weeks fit the priors, last two evaluate them

    # Drone motion and energy
    v_xy: float = 1.0           # m/s horizontal cruise speed indoors
    v_z: float = 0.5            # m/s climb/descent speed
    observe_time_s: float = 2.0  # hover-and-detect time at each viewpoint
    hover_power_w: float = 180.0
    mass_kg: float = 1.5
    climb_efficiency: float = 0.5
    energy_weight: float = 0.002  # lambda: seconds of cost per joule (cost = time + lambda * energy)
    time_budget_s: float = 240.0

    # Viewpoints
    altitudes: tuple = (0.4, 1.0, 1.8)
    fixed_altitude: float = 0.4   # ground-robot-equivalent ablation
    cruise_altitude: float = 1.0
    vantage_spacing: float = 1.0
    grid_res: float = 0.25
    path_inflation: float = 0.25
    approach_ring: tuple = (0.35, 1.0)  # approach point distance from the object (success radius is 1 m)

    # Perception
    max_range: float = 7.0
    object_height: float = 0.2    # target point above the supporting surface
    reid_threshold: float = 0.93
    obs_noise: float = 0.05

    # Experiments
    horizons_h: tuple = (1.0, 24.0)
    n_commands_per_horizon: int = 250
    n_grounding: int = 600
    conformal_alpha: float = 0.1
    n_boot: int = 1000

    @classmethod
    def quick(cls, **kw):
        """Small settings for tests and smoke runs."""
        base = dict(days=21, train_days=14, n_commands_per_horizon=30, n_grounding=200, n_boot=200)
        base.update(kw)
        return cls(**base)
