"""Simulation configuration. All times are in hours unless the name ends in `_s` (seconds)."""
from dataclasses import dataclass


@dataclass
class SimConfig:
    seed: int = 0
    layout: str = "open"       # 'open' | 'cubicle' | 'booth' (see world.make_world)
    partition_h: float = None  # partition / cubicle-wall height in m (None = layout default)
    clutter_h: float = 0.0     # height of a block of clutter in the middle of every desk (0 = none)
    n_chairs: int = 12         # number of identical-looking chairs
    identical_classes: tuple = ("chair",)  # classes whose instances look the same
    identical_sigma: float = 0.02          # appearance spread among identical instances

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
    visibility_model: str = "point"  # 'point' (Stages 1-2) | 'extended' (calibrated on Isaac Sim renders)
    vis_a50_px: float = 200.0        # extended model: pixel area for half-maximal detection
    vis_slope: float = 0.35          # extended model: logistic width in ln(pixels)
    vis_pmax: float = 0.95           # extended model: detection probability for very large objects
    laptop_open: bool = False        # extended model: laptops open (screen up, ~0.25 m tall) instead of closed (3 cm)

    # Experiments
    horizons_h: tuple = (1.0, 24.0)
    n_commands_per_horizon: int = 250
    n_grounding: int = 600
    conformal_alpha: float = 0.1
    n_boot: int = 1000

    def world_kwargs(self):
        return dict(layout=self.layout, partition_h=self.partition_h, clutter_h=self.clutter_h,
                    n_chairs=self.n_chairs, identical_classes=self.identical_classes,
                    identical_sigma=self.identical_sigma)

    def drone_key(self):
        """Everything the precomputed drone matrices depend on (not the seed, history or perception)."""
        return (self.layout, self.partition_h, self.clutter_h, self.v_xy, self.v_z, self.hover_power_w,
                self.mass_kg, self.climb_efficiency, self.energy_weight, self.observe_time_s, self.altitudes,
                self.fixed_altitude, self.cruise_altitude, self.vantage_spacing, self.grid_res,
                self.path_inflation, self.approach_ring, self.max_range, self.object_height,
                self.visibility_model, self.vis_a50_px, self.vis_slope, self.vis_pmax, self.laptop_open)

    @classmethod
    def quick(cls, **kw):
        """Small settings for tests and smoke runs."""
        base = dict(days=21, train_days=14, n_commands_per_horizon=30, n_grounding=200, n_boot=200)
        base.update(kw)
        return cls(**base)
