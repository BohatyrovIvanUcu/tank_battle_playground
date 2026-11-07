import math
import random
from typing import List, Dict, Any

class RandomBot:
    """Random baseline bot taking continuous actions."""
    def act(self, obs: Dict[str, Any]) -> List[float]:
        return [
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
        ]


class RuleBasedBot:
    """
    Heuristic rule-based bot:
    - Steers body to keep medium distance (around 250 units) from enemy.
    - Continuously aims turret at enemy angle.
    - Fires whenever aligned within 15 degrees.
    """
    def __init__(self, target_dist: float = 0.35):
        self.target_dist = target_dist

    def act(self, obs: Dict[str, Any]) -> List[float]:
        dist = obs.get("enemy_dist", 0.5)
        angle_to_enemy = obs.get("enemy_angle_to", 0.0)
        turret_angle = obs.get("turret_angle", 0.0)
        body_angle = obs.get("body_angle", 0.0)

        # 1. Body steering: turn towards enemy
        turn = math.sin(angle_to_enemy)
        turn = max(-1.0, min(1.0, turn * 2.0))

        # 2. Forward/Backward thrust based on desired distance
        if dist > self.target_dist + 0.05:
            thrust = 0.8
        elif dist < self.target_dist - 0.05:
            thrust = -0.5
        else:
            thrust = 0.2

        # 3. Turret steering: calculate difference between turret angle and angle to enemy
        world_angle_to_enemy = body_angle + angle_to_enemy
        turret_error = math.sin(world_angle_to_enemy - turret_angle)
        turret_turn = max(-1.0, min(1.0, turret_error * 3.0))

        # 4. Shooting: shoot if turret is pointed near enemy (< 0.2 rad ~ 11 deg)
        shoot = 1.0 if abs(turret_error) < 0.25 else -1.0

        return [thrust, turn, turret_turn, shoot]
