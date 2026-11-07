"""
=============================================================================
🤝 FRIEND / COLLABORATOR TEMPLATE: Custom Neural Network & Inputs
=============================================================================
This template demonstrates how you and your collaborator can develop
conceptually different neural network architectures with completely custom
inputs, while interacting with the exact same 2D Tank Battle Playground!

CONTRACT:
1. INPUTS: You can extract whatever features you want from the raw telemetry!
2. OUTPUTS: You MUST produce 4 continuous values in [-1.0, 1.0]:
   - action[0]: thrust (-1.0 reverse to 1.0 full forward)
   - action[1]: turn (-1.0 steer left to 1.0 steer right)
   - action[2]: turret_turn (-1.0 aim left to 1.0 aim right)
   - action[3]: shoot (> 0.5 triggers cannon shot when cooldown ready)
=============================================================================
"""

import math
import numpy as np
from tank_env import TankEnvironment, ObservationAdapter, SimpleMLP, RuleBasedBot

# ---------------------------------------------------------------------------
# STEP 1: Define YOUR Custom Observation Adapter (Your chosen input features)
# ---------------------------------------------------------------------------
class FriendObservationAdapter(ObservationAdapter):
    """
    Example of a completely custom input adapter.
    Instead of raw cartesian coords, this extracts:
    - Distance to enemy
    - Relative angle to enemy (in radians normalized by pi)
    - Relative enemy turret angle
    - Tank health ratio (0.0 to 1.0)
    - Enemy health ratio (0.0 to 1.0)
    - Reload readiness (0.0 = ready, 1.0 = cooling down)
    - Line of Sight (LOS): is enemy directly visible or behind a wall? (1.0 = visible, 0.0 = hidden)
    - 4 Cardinal Lidar distance sensors (front, right, back, left)
    - Closest incoming enemy bullet distance and angle
    Total dimensions: 12
    """
    @property
    def output_dim(self) -> int:
        return 12

    def extract(self, raw: dict) -> np.ndarray:
        # 1. Combat basics
        dist = raw.get("enemy_dist", 0.5)
        angle_to = raw.get("enemy_angle_to", 0.0) / math.pi
        enemy_turret = raw.get("enemy_turret_angle", 0.0) / math.pi
        hp = raw.get("hp", 100.0) / 100.0
        enemy_hp = raw.get("enemy_hp", 100.0) / 100.0
        reload_cd = raw.get("reload_cooldown", 0.0)
        enemy_visible = 1.0 if raw.get("enemy_visible", True) else 0.0

        # 2. 4 Lidar directions: Front (ray 3), Right (ray 5), Back (ray 7), Left (ray 1)
        lidar = raw.get("lidar_rays", [1.0] * 8)
        front = lidar[3] if len(lidar) > 3 else 1.0
        right = lidar[5] if len(lidar) > 5 else 1.0
        back = lidar[7] if len(lidar) > 7 else 1.0
        left = lidar[1] if len(lidar) > 1 else 1.0

        # 3. Threat detection: nearest incoming bullet
        bullets = raw.get("bullets", [])
        nearest_bullet_dist = 1.0
        for b in bullets:
            # bullet format: [rel_x, rel_y, rel_vx, rel_vy, is_enemy]
            if len(b) >= 5 and b[4] == 1.0: # enemy bullet
                b_dist = math.hypot(b[0], b[1])
                if b_dist < nearest_bullet_dist:
                    nearest_bullet_dist = b_dist

        features = [
            dist,
            angle_to,
            enemy_turret,
            hp,
            enemy_hp,
            reload_cd,
            enemy_visible,
            front,
            right,
            back,
            left,
            nearest_bullet_dist,
        ]
        return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# STEP 2: Running a Quick Match using your Custom Adapter
# ---------------------------------------------------------------------------
def main():
    print("Initializing Tank Playground with Friend's Custom Input Adapter...")
    env = TankEnvironment(max_steps=1000)
    adapter = FriendObservationAdapter()

    # Create your model (or load from friend_model.npz)
    my_model = SimpleMLP(
        input_dim=adapter.output_dim,
        hidden_dims=[32, 16],
        output_dim=4,
    )

    opponent_bot = RuleBasedBot()

    obs1_raw, obs2_raw = env.reset(seed=123)
    done = False
    step = 0

    print(f"Model initialized with input dimension: {adapter.output_dim}")
    print("Stepping simulation...")

    while not done and step < 200:
        step += 1

        # Tank 1 uses your custom adapter
        x1 = adapter.extract(obs1_raw)
        action1 = my_model.forward(x1)

        # Tank 2 is a rule-based opponent
        action2 = opponent_bot.act(obs2_raw)

        obs1_raw, obs2_raw, rewards, done, info = env.step(action1, action2)

    print(f"Simulation test passed! Completed {step} ticks successfully.")
    print(f"Tank 1 HP: {obs1_raw['hp']} | Tank 2 HP: {obs2_raw['hp']}")
    print("You are ready to train your model using python/train_ga.py or custom PyTorch loop!")

if __name__ == "__main__":
    main()
