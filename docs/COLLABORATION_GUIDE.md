# 🤝 Collaboration Guide: Tank Battle Playground

This guide outlines how you and your collaborator can train separate, competing neural network models on this shared playground and host sparring matches between them.

---

## 🚀 1. Setup in 3 Commands

Ensure you have Rust and Python 3.10+ installed:

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install build and training requirements
pip install maturin pytest numpy

# 3. Build and install the Rust simulation engine
maturin develop --release
```

---

## 🧠 2. The Core Idea: Freedom in Inputs, Uniform in Control

You and your friend can train **conceptually completely different AI models**:
- **Player A** can use 8-direction lidar raycasts to dodge walls and bullets.
- **Player B** can use polar coordinates (distance, angles) and bullet trajectory vectors.
- One can use a Genetic Algorithm with MLP (`python/tank_env/neural_net.py`), while the other can use PyTorch, PPO, or NEAT.

### The Universal Contract (Last Layer):
Every agent must return 4 continuous floats bounded in `[-1.0, 1.0]`:
| Index | Action | Range | Description |
|---|---|---|---|
| `action[0]` | `thrust` | `[-1.0 .. 1.0]` | Forward acceleration (>0) / Reverse (<0) |
| `action[1]` | `turn` | `[-1.0 .. 1.0]` | Hull steering: Left (<0) / Right (>0) |
| `action[2]` | `turret_turn` | `[-1.0 .. 1.0]` | Turret aiming: Left (<0) / Right (>0) |
| `action[3]` | `shoot` | `[-1.0 .. 1.0]` | Fire cannon if `> 0.5` and cooldown is ready |

---

## ⚔️ 3. Training Your Model

Run genetic algorithm training in headless mode (takes only seconds for multiple generations):

```bash
# Train against the heuristic rule-based bot
PYTHONPATH=python python3 python/train_ga.py --generations 25 --population 30 --adapter vector

# Checkpoints will be saved to checkpoints/best_model.npz
```

---

## 🏆 4. Sparring: You vs Your Friend's Model

Once each of you has trained a model (saved as a `.npz` file):

```bash
# Run a 3-match duel rendered in real-time at 60 FPS:
PYTHONPATH=python python3 python/watch_match.py --tank1 checkpoints/best_model.npz --tank2 path/to/friend_model.npz
```

Tanks are rendered in contrasting neon colors:
- **Tank 1 (Player 1)**: Cyan with white turret line and green health bar.
- **Tank 2 (Player 2)**: Bright Coral/Red with white turret line and orange health bar.
- **Top HUD**: Shows health bars, current tick progress, and bullet trajectories.
