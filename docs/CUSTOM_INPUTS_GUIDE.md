# 📊 Custom Inputs & Observation Guide

The Rust engine emits rich physical telemetry at every tick. You can inspect or transform any combination of these properties for your neural network.

---

## 📡 Raw Telemetry Dictionary Schema

When `obs1, obs2 = env.step(...)` or `env.reset()` is called, each observation dictionary contains:

| Key | Type | Description |
|---|---|---|
| `x` | `float` in `[0, 1]` | Normalized X position in arena (`pos.x / width`) |
| `y` | `float` in `[0, 1]` | Normalized Y position in arena (`pos.y / height`) |
| `vx` | `float` in `[-1, 1]` | Normalized X velocity |
| `vy` | `float` in `[-1, 1]` | Normalized Y velocity |
| `speed` | `float` in `[-1, 1]` | Forward/reverse linear speed |
| `body_angle` | `float` in `[-π, π]` | Hull rotation in radians |
| `turret_angle` | `float` in `[-π, π]` | Turret aiming angle in radians |
| `hp` | `float` in `[0, 100]` | Current Health Points (initial 100) |
| `reload_cooldown`| `float` in `[0, 1]` | Fraction of reload timer remaining (0.0 = ready to fire) |
| `enemy_rel_x` | `float` in `[-1, 1]` | Relative X distance to enemy (`(enemy.x - own.x) / width`) |
| `enemy_rel_y` | `float` in `[-1, 1]` | Relative Y distance to enemy (`(enemy.y - own.y) / height`) |
| `enemy_dist` | `float` in `[0, 1]` | Normalized Euclidean distance to enemy |
| `enemy_angle_to`| `float` in `[-π, π]` | Relative angle pointing directly at enemy from hull heading |
| `enemy_turret_angle`| `float` in `[-π, π]` | Enemy turret direction |
| `enemy_hp` | `float` in `[0, 100]` | Enemy current HP |
| `enemy_visible` | `bool` (`True`/`False`) | **Line of Sight (LOS)**: `True` if direct line of sight to enemy; `False` if blocked by a wall |
| `lidar_rays` | `list[float]` (8 elements) | Distance to nearest obstacle (wall, obstacle, or enemy) along 8 rays |
| `bullets` | `list[list[float]]` | Active bullets: `[rel_x, rel_y, rel_vx, rel_vy, is_enemy]` |

---

## 🧱 Obstacles & Partition Walls

You can place impenetrable walls in the arena that block both tank movement, projectiles, and direct line of sight:

```python
env = TankEnvironment()
# Add a vertical partition wall in the center
env.add_obstacle(x=385.0, y=150.0, width=30.0, height=300.0)

# Clear obstacles if needed
env.clear_obstacles()
```

---

## 🛠 Creating Your Custom Adapter

Inherit from `ObservationAdapter` and implement `extract(self, raw: dict) -> np.ndarray`:

```python
from tank_env import ObservationAdapter
import numpy as np

class PolarObservationAdapter(ObservationAdapter):
    @property
    def output_dim(self) -> int:
        return 6

    def extract(self, raw: dict) -> np.ndarray:
        return np.array([
            raw["enemy_dist"],
            raw["enemy_angle_to"] / 3.14159,
            raw["hp"] / 100.0,
            raw["enemy_hp"] / 100.0,
            raw["reload_cooldown"],
            raw["speed"],
        ], dtype=np.float32)
```

See [examples/friend_template.py](file:///home/ivan/genitic_algorithm_test/examples/friend_template.py) for an extended example with bullet dodging sensors.
