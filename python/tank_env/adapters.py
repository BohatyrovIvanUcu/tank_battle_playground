from abc import ABC, abstractmethod
import math
import numpy as np

class ObservationAdapter(ABC):
    """
    Abstract adapter for transforming raw telemetry from TankArena
    into a tensor/vector suitable for a specific neural network architecture.
    """
    @abstractmethod
    def extract(self, raw: dict) -> np.ndarray:
        """Takes raw telemetry dictionary and outputs numpy array."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Returns input feature dimension for neural network."""
        pass


class VectorObservationAdapter(ObservationAdapter):
    """
    Standard 14-dimensional normalized feature vector:
    - Own position: x, y in [0, 1]
    - Own velocity: vx, vy in [-1, 1]
    - Own forward speed: speed in [-1, 1]
    - Own body angle: angle / pi in [-1, 1]
    - Own turret angle: turret_angle / pi in [-1, 1]
    - Own HP: hp / 100.0 in [0, 1]
    - Own reload timer: reload_cooldown in [0, 1]
    - Enemy relative position: rel_x, rel_y in [-1, 1]
    - Enemy distance: dist in [0, 1]
    - Angle to enemy: angle_to / pi in [-1, 1]
    - Enemy HP: enemy_hp / 100.0 in [0, 1]
    """
    @property
    def output_dim(self) -> int:
        return 14

    def extract(self, raw: dict) -> np.ndarray:
        features = [
            raw.get("x", 0.0),
            raw.get("y", 0.0),
            raw.get("vx", 0.0),
            raw.get("vy", 0.0),
            raw.get("speed", 0.0),
            raw.get("body_angle", 0.0) / math.pi,
            raw.get("turret_angle", 0.0) / math.pi,
            raw.get("hp", 100.0) / 100.0,
            raw.get("reload_cooldown", 0.0),
            raw.get("enemy_rel_x", 0.0),
            raw.get("enemy_rel_y", 0.0),
            raw.get("enemy_dist", 0.0),
            raw.get("enemy_angle_to", 0.0) / math.pi,
            raw.get("enemy_hp", 100.0) / 100.0,
        ]
        return np.array(features, dtype=np.float32)


class LidarObservationAdapter(ObservationAdapter):
    """
    Lidar-based observation vector:
    - 8 directional raycast distances: [0, 1]
    - 4 tank state scalars: speed, hp / 100, reload_cooldown, enemy_angle_to / pi
    Total dimension = 12
    """
    def __init__(self, num_rays: int = 8):
        self.num_rays = num_rays

    @property
    def output_dim(self) -> int:
        return self.num_rays + 4

    def extract(self, raw: dict) -> np.ndarray:
        rays = raw.get("lidar_rays", [1.0] * self.num_rays)
        if len(rays) > self.num_rays:
            rays = rays[:self.num_rays]
        elif len(rays) < self.num_rays:
            rays = list(rays) + [1.0] * (self.num_rays - len(rays))

        state_features = [
            raw.get("speed", 0.0),
            raw.get("hp", 100.0) / 100.0,
            raw.get("reload_cooldown", 0.0),
            raw.get("enemy_angle_to", 0.0) / math.pi,
        ]
        return np.array(list(rays) + state_features, dtype=np.float32)
