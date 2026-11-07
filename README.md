# 🛡️ 2D Tank Battle Playground & Neuroevolution Platform

A high-performance 2D tank combat simulation engine built in **Rust** with **PyO3** Python bindings and native 60 FPS visual rendering.

Designed for competitive multi-agent training: you and your collaborator can design conceptually different neural network architectures with custom observation spaces while maintaining a unified continuous control contract.

---

## ⚡ Key Highlights

- **Blazing Fast Simulation**: Physics, raycasting, and ballistics run in deterministic Rust at over 10,000 steps/sec.
- **Native 2D Visualizer**: Real-time 60 FPS graphics window powered by `minifb` (zero heavy system C-dependencies, pure pixel framebuffer).
- **Uniform Control Contract**: Standardized `[-1.0 .. 1.0]` continuous action space:
  - `thrust` (forward/reverse)
  - `turn` (hull steering)
  - `turret_turn` (cannon aiming)
  - `shoot` (fire cannon when cooldown ready)
- **Modular Observation Adapters**: Clean separation between raw physical telemetry and network inputs. Customize what your neural net sees without touching the simulation!
- **Asynchronous & Lockstep Support**: Environment supports both lockstep `step(a1, a2)` and asynchronous parallel execution where one agent acts while the other is thinking.
- **Genetic Algorithm Suite**: Built-in pure NumPy feedforward network (`SimpleMLP`), population manager, and checkpoint system (`train_ga.py`).

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release
```

### 2. Run Instant Visual Match (Rust Native)
To verify the visualizer without training:
```bash
cargo run --release --bin standalone_viewer
```

### 3. Train a Model with Genetic Algorithm
```bash
PYTHONPATH=python python3 python/train_ga.py --generations 20 --population 30 --adapter vector
```
Checkpoints will be saved to `checkpoints/best_model.npz`.

### 4. Watch Matches in Real-Time Window
```bash
# Watch your trained model fight the rule-based bot:
PYTHONPATH=python python3 python/watch_match.py --tank1 checkpoints/best_model.npz --tank2 rule_based

# Watch your model duel your friend's model:
PYTHONPATH=python python3 python/watch_match.py --tank1 checkpoints/best_model.npz --tank2 friend_model.npz
```

---

## 📂 Project Structure

```
├── crates/
│   ├── tank_core/          # Core Rust physics, raycast math, and minifb renderer
│   │   ├── src/arena.rs    # Game state, collision resolution, damage, HP
│   │   ├── src/math.rs     # 2D vector algebra, ray-circle & ray-box intersections
│   │   ├── src/renderer.rs # 2D software rasterizer & window renderer
│   │   └── src/bin/        # Standalone binary runner
│   └── tank_pyo3/          # PyO3 C-extension exposing PyTankArena to Python
├── python/
│   ├── tank_env/           # Python SDK wrapper
│   │   ├── adapters.py     # Modular observation extractors (Vector, Lidar)
│   │   ├── environment.py  # Gym-like TankEnvironment
│   │   ├── neural_net.py   # SimpleMLP with chromosome pack/unpack for GA
│   │   └── bots.py         # Heuristic & random baseline bots
│   ├── train_ga.py         # Genetic algorithm training pipeline
│   └── watch_match.py      # Real-time visual match runner
├── examples/
│   └── friend_template.py  # Copy-paste starting template for your collaborator
├── docs/
│   ├── COLLABORATION_GUIDE.md  # Multi-developer workflow guide
│   └── CUSTOM_INPUTS_GUIDE.md  # Full telemetry dictionary documentation
└── tests/                  # Rust and Python integration test suite
```

---

## 🧪 Testing

Run Rust tests:
```bash
cargo test --all
```

Run Python tests:
```bash
source .venv/bin/activate
PYTHONPATH=python pytest tests/
```
