# Tank 2D Battle Playground & Genetic Algorithm Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-performance 2D tank battle simulation playground in Rust with PyO3 Python bindings, native visualizer, uniform action space, modular observation extractor, and a complete genetic algorithm training workflow shareable with collaborators.

**Architecture:** A deterministic 2D physics simulation core in Rust (`tank_core`) executes tank movement, turret aiming, projectile ballistics, and collisions at maximum FPS. A native 2D window renderer (`minifb`) enables real-time visual inspection without Python GUI overhead. PyO3 exposes a clean Gymnasium-style 2-agent interface (`reset()`, `step(action_t1, action_t2)`, `render()`) to Python. On top, a Python layer provides customizable observation adapters so two developers can attach conceptually different neural network architectures (e.g. raycast lidar vs feature vectors) while preserving a standardized `[-1.0, 1.0]` continuous action space.

**Tech Stack:** Rust (1.99+), PyO3 (0.24+), Maturin, Minifb (software framebuffer 2D rendering), Python (3.12+), NumPy, PyTorch (or pure NumPy NN for GA).

**Spec:** User requirements from `/grill-me` interview:
- 2 tanks with survival and combat goals (destroy opponent, dodge bullets, preserve HP).
- 100 HP each, firing cooldown, limited projectile speed, open arena with boundary walls, step limit (1500 steps) to prevent stalemates.
- Continuous action space `[-1.0 .. 1.0]` for `[thrust, turn, turret_turn, shoot]`.
- Built-in native window visualizer runnable from Rust and controllable from Python.
- Multi-agent step API `step(action_tank1, action_tank2)`.
- Modular observation system allowing developer and friend to craft independent neural network input encodings.

## Global Constraints

- Engine core must have zero heap allocation during `step()` for peak simulation throughput.
- Simulation must be deterministic given identical initial seeds and action sequences.
- Native visualizer must run without requiring complex system dependencies (pure software buffer via `minifb`).
- Python package must compile cleanly using `maturin develop` inside a standard Python venv.
- Code and identifiers in English; user reports and documentation in Ukrainian/English.

## Review Focus

- Tanks clipping through walls when high thrust is applied at arena boundaries: clamp positions and zero out normal velocity.
- Bullets tunneling through opposing tank if bullet speed is greater than tank collision radius: ray-circle segment intersection testing.
- Rapid firing exploit without cooldown: enforce strict tick countdown before a tank can shoot again.
- PyO3 GIL deadlocks when running headless training loops: release GIL (`py.allow_threads`) during batch simulation steps.
- Numerical instability or NaN actions from neural networks: clamp all incoming actions strictly to `[-1.0, 1.0]` before physics integration.

---

### Task 1: Rust Core Foundation & Math (`tank_core`)

**Files:**
- Create: `Cargo.toml`
- Create: `pyproject.toml`
- Create: `crates/tank_core/Cargo.toml`
- Create: `crates/tank_core/src/lib.rs`
- Create: `crates/tank_core/src/types.rs`
- Create: `crates/tank_core/src/math.rs`
- Test: `crates/tank_core/tests/test_math.rs`

**Interfaces:**
- Consumes: None (root module)
- Produces: `Vec2`, `Ray`, `TankAction`, `TankState`, `BulletState`, `ArenaConfig`, `MatchResult`

- [ ] **Step 1: Write the failing test for math & vector helpers**

```rust
// crates/tank_core/tests/test_math.rs
use tank_core::math::{Vec2, ray_circle_intersect};

#[test]
fn test_vec2_operations() {
    let a = Vec2::new(3.0, 4.0);
    assert!((a.length() - 5.0).abs() < 1e-5);
    let n = a.normalize();
    assert!((n.length() - 1.0).abs() < 1e-5);
}

#[test]
fn test_ray_circle_intersection() {
    let ray_origin = Vec2::new(0.0, 0.0);
    let ray_dir = Vec2::new(1.0, 0.0);
    let circle_center = Vec2::new(10.0, 0.0);
    let radius = 2.0;
    let hit = ray_circle_intersect(ray_origin, ray_dir, circle_center, radius);
    assert!(hit.is_some());
    let (t, _hit_point) = hit.unwrap();
    assert!((t - 8.0).abs() < 1e-4);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package tank_core --test test_math`
Expected: FAIL (crate or files not found)

- [ ] **Step 3: Implement `Vec2`, `math.rs`, and core data types in `crates/tank_core/src/`**

Define `Vec2`, basic 2D vector algebra, raycast segment intersection, `ArenaConfig` (width, height, max_steps, tank_radius, bullet_speed, reload_ticks), `TankAction` (thrust, turn, turret_turn, shoot), and `TankState` (pos, vel, body_angle, turret_angle, hp, reload_timer, score).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --package tank_core --test test_math`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml pyproject.toml crates/tank_core
git commit -m "feat(core): add tank_core math, types and test suite"
```

---

### Task 2: Simulation Physics & Arena Logic (`tank_core::arena`)

**Files:**
- Create: `crates/tank_core/src/arena.rs`
- Modify: `crates/tank_core/src/lib.rs`
- Test: `crates/tank_core/tests/test_arena.rs`

**Interfaces:**
- Consumes: `Vec2`, `TankAction`, `TankState`, `ArenaConfig` from Task 1
- Produces: `TankArena`, `StepOutcome`, `TankTelemetry`

- [ ] **Step 1: Write failing tests for arena mechanics**

```rust
// crates/tank_core/tests/test_arena.rs
use tank_core::arena::TankArena;
use tank_core::types::{ArenaConfig, TankAction};

#[test]
fn test_arena_initialization_and_step() {
    let config = ArenaConfig::default();
    let mut arena = TankArena::new(config, 42);
    let (t1, t2) = arena.get_states();
    assert_eq!(t1.hp, 100.0);
    assert_eq!(t2.hp, 100.0);

    let action = TankAction { thrust: 1.0, turn: 0.0, turret_turn: 0.0, shoot: false };
    let outcome = arena.step(action, action);
    assert_eq!(outcome.step_count, 1);
    assert!(!outcome.done);
}

#[test]
fn test_bullet_damage_and_cooldown() {
    let mut config = ArenaConfig::default();
    config.reload_ticks = 10;
    config.bullet_damage = 25.0;
    let mut arena = TankArena::new(config, 42);
    // Face tank 1 toward tank 2 and fire
    arena.set_positions_for_testing(100.0, 100.0, 0.0, 200.0, 100.0, std::f32::consts::PI);
    let shoot = TankAction { thrust: 0.0, turn: 0.0, turret_turn: 0.0, shoot: true };
    let noop = TankAction::default();
    arena.step(shoot, noop);
    assert_eq!(arena.bullets().len(), 1);

    // Simulate flight until impact
    for _ in 0..30 {
        arena.step(noop, noop);
    }
    let (_t1, t2) = arena.get_states();
    assert_eq!(t2.hp, 75.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package tank_core --test test_arena`
Expected: FAIL (`TankArena` not defined)

- [ ] **Step 3: Implement `TankArena` in `crates/tank_core/src/arena.rs`**

Implement deterministic step logic:
- Apply linear acceleration with drag and friction.
- Angular integration for body and turret.
- Boundary collision resolution: clamp tank position within `[radius, arena_size - radius]` and zero wall-normal velocity.
- Tank-to-tank elastic / repulsive circle collision.
- Bullet generation with cooldown decrement.
- Bullet ballistics and continuous ray-circle collision detection against both tanks and arena perimeter walls.
- Damage deduction (25 HP per hit, 100 HP initial = 4 hits to destroy).
- Reward signal calculation (hit dealt, hit taken, survival tick bonus, distance penalty/reward).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --package tank_core --test test_arena`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/tank_core/src/arena.rs crates/tank_core/src/lib.rs crates/tank_core/tests/test_arena.rs
git commit -m "feat(core): implement tank physics, bullet ballistics, and collisions"
```

---

### Task 3: Built-in Native Rust Visualizer (`minifb`)

**Files:**
- Modify: `crates/tank_core/Cargo.toml`
- Create: `crates/tank_core/src/renderer.rs`
- Create: `crates/tank_core/src/bin/standalone_viewer.rs`
- Test: `crates/tank_core/tests/test_renderer.rs`

**Interfaces:**
- Consumes: `TankArena`, `TankState`, `BulletState`, `ArenaConfig` from Tasks 1-2
- Produces: `ArenaRenderer` with methods `new(width, height)`, `render_frame(&arena)`, `is_open() -> bool`

- [ ] **Step 1: Write headless frame buffer test for the renderer**

```rust
// crates/tank_core/tests/test_renderer.rs
use tank_core::arena::TankArena;
use tank_core::renderer::ArenaRenderer;
use tank_core::types::ArenaConfig;

#[test]
fn test_renderer_buffer_generation() {
    let arena = TankArena::new(ArenaConfig::default(), 42);
    let mut renderer = ArenaRenderer::new_headless(800, 600);
    renderer.draw_arena(&arena);
    let buf = renderer.buffer();
    assert_eq!(buf.len(), 800 * 600);
    // Ensure screen is not completely blank (black)
    assert!(buf.iter().any(|&p| p != 0x000000));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package tank_core --test test_renderer`
Expected: FAIL (`ArenaRenderer` not defined)

- [ ] **Step 3: Implement `ArenaRenderer` with `minifb` support**

- Software rasterizer for high-contrast arena rendering:
  - Arena background and boundary borders.
  - Tank bodies (directional rectangles or polygons with heading line).
  - Turret barrels pointing in turret aiming angle.
  - Distinct colors for Tank 1 (Cyan/Green) and Tank 2 (Orange/Red).
  - Health bars above each tank (green/yellow/red depending on current HP).
  - Glowing bullets with brief trail effect.
  - HUD header: current tick, HP stats, score counters.
- Provide both `render_window()` (interactive GUI window via `minifb` running at 60 FPS) and `new_headless()` (in-memory buffer for headless testing or export).
- Create `src/bin/standalone_viewer.rs` so users can run `cargo run --release --bin standalone_viewer` to watch two demo bots duel right away.

- [ ] **Step 4: Run test and build binary**

Run: `cargo test --package tank_core --test test_renderer`
Expected: PASS
Run: `cargo check --bin standalone_viewer`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/tank_core/src/renderer.rs crates/tank_core/src/bin/standalone_viewer.rs crates/tank_core/tests/test_renderer.rs
git commit -m "feat(viewer): implement native minifb 2D visualizer and standalone runner"
```

---

### Task 4: PyO3 Python Bindings (`tank_arena` Extension)

**Files:**
- Create: `crates/tank_pyo3/Cargo.toml`
- Create: `crates/tank_pyo3/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)
- Test: `tests/test_pyo3_bindings.py`

**Interfaces:**
- Consumes: `tank_core::arena::TankArena`, `ArenaRenderer`
- Produces: Python C-extension `tank_arena` exposing class `PyTankArena` with methods:
  - `reset(seed: Optional[int]) -> tuple[dict, dict]`
  - `step(action_t1: list[float], action_t2: list[float]) -> tuple[dict, dict, list[float], bool, dict]`
  - `render() -> bool`
  - `close()`

- [ ] **Step 1: Write python test verifying the PyO3 module interface**

```python
# tests/test_pyo3_bindings.py
import pytest
import tank_arena

def test_arena_lifecycle():
    arena = tank_arena.PyTankArena(width=800.0, height=600.0, max_steps=500, seed=123)
    obs1, obs2 = arena.reset()
    assert "x" in obs1 and "y" in obs1
    assert "hp" in obs1 and obs1["hp"] == 100.0
    assert "enemy_dist" in obs1

    act1 = [1.0, 0.0, 0.0, 0.0]
    act2 = [0.0, 1.0, 0.0, 1.0]
    next_obs1, next_obs2, rewards, done, info = arena.step(act1, act2)
    assert len(rewards) == 2
    assert not done
    assert info["step"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pyo3_bindings.py`
Expected: FAIL (`ModuleNotFoundError: No module named 'tank_arena'`)

- [ ] **Step 3: Implement PyO3 wrapper in `crates/tank_pyo3/src/lib.rs`**

- Wrap `TankArena` in `PyTankArena` struct.
- Expose telemetry dictionaries with keys:
  - `x`, `y`, `vx`, `vy`, `angle`, `turret_angle`, `hp`, `reload_cooldown`
  - `enemy_rel_x`, `enemy_rel_y`, `enemy_dist`, `enemy_angle_to`, `enemy_hp`
  - `rays`: 8 raycasts (lidar) detecting distance to walls and enemy tank
  - `bullets`: list of visible active bullets with relative coordinates and velocities.
- Add `render()` binding calling `ArenaRenderer` to display interactive window.
- Release Python GIL during computation.

- [ ] **Step 4: Build wheel with maturin and run test**

Run: `maturin develop --release && pytest tests/test_pyo3_bindings.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/tank_pyo3 Cargo.toml tests/test_pyo3_bindings.py
git commit -m "feat(pyo3): create PyO3 Python bindings for TankArena"
```

---

### Task 5: Python SDK Layer & Observation Adapters

**Files:**
- Create: `python/tank_env/__init__.py`
- Create: `python/tank_env/environment.py`
- Create: `python/tank_env/adapters.py`
- Create: `python/tank_env/bots.py`
- Test: `tests/test_adapters.py`

**Interfaces:**
- Consumes: `tank_arena.PyTankArena`
- Produces:
  - `TankEnvironment`: High-level wrapper with gym-like API.
  - `ObservationAdapter` (Base class): allows user & friend to customize input shapes.
  - `VectorObservationAdapter`: Standard 16-element float array for simple MLP networks.
  - `LidarObservationAdapter`: Raycast distances + relative enemy vector.
  - Baseline bots: `RandomBot`, `RuleBasedBot` (aims turret at opponent and fires).

- [ ] **Step 1: Write test for observation adapters and baseline bots**

```python
# tests/test_adapters.py
import numpy as np
from tank_env.environment import TankEnvironment
from tank_env.adapters import VectorObservationAdapter, LidarObservationAdapter
from tank_env.bots import RuleBasedBot

def test_observation_adapters():
    env = TankEnvironment(seed=99)
    obs1_raw, obs2_raw = env.reset()

    vec_adapter = VectorObservationAdapter()
    v1 = vec_adapter.extract(obs1_raw)
    assert isinstance(v1, np.ndarray)
    assert v1.shape == (14,)
    assert not np.isnan(v1).any()

    lidar_adapter = LidarObservationAdapter(num_rays=8)
    l1 = lidar_adapter.extract(obs1_raw)
    assert isinstance(l1, np.ndarray)
    assert l1.shape == (12,) # 8 rays + 4 state floats

def test_rule_based_bot():
    env = TankEnvironment(seed=99)
    bot = RuleBasedBot()
    obs1, obs2 = env.reset()
    action = bot.act(obs1)
    assert len(action) == 4
    assert all(-1.0 <= a <= 1.0 for a in action)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapters.py`
Expected: FAIL (modules not found)

- [ ] **Step 3: Implement Python SDK, adapters, and baseline bots**

- Write `TankEnvironment` wrapping `PyTankArena` with clean exception handling and configuration.
- Implement abstract `ObservationAdapter` and default implementations (`VectorObservationAdapter`, `LidarObservationAdapter`).
- Implement `RuleBasedBot` (heuristic tank controller that rotates towards the enemy, drives forward if far, and shoots when aligned).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapters.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/tank_env tests/test_adapters.py
git commit -m "feat(sdk): add Python SDK, modular observation adapters and rule-based bot"
```

---

### Task 6: Genetic Algorithm Training Pipeline & Visual Match Runner

**Files:**
- Create: `python/train_ga.py`
- Create: `python/watch_match.py`
- Create: `python/tank_env/neural_net.py`
- Test: `tests/test_ga_pipeline.py`

**Interfaces:**
- Consumes: `TankEnvironment`, `ObservationAdapter`, `RuleBasedBot` from Task 5
- Produces:
  - `SimpleMLP`: Fast NumPy-based feedforward neural network (weights represented as flat 1D chromosome for simple GA crossover and mutation).
  - `GAPopulation`: Population manager with tournament selection, elitism, mutation, and crossover.
  - `train_ga.py`: Standalone CLI script to train tanks against bot or via co-evolution.
  - `watch_match.py`: Script to load saved tank weights and watch live duels with the native visualizer.

- [ ] **Step 1: Write test for neural network and GA step execution**

```python
# tests/test_ga_pipeline.py
import numpy as np
from tank_env.neural_net import SimpleMLP
from tank_env.adapters import VectorObservationAdapter
from tank_env.environment import TankEnvironment

def test_mlp_forward_and_chromosome():
    mlp = SimpleMLP(input_dim=14, hidden_dims=[16, 16], output_dim=4)
    chrom = mlp.get_chromosome()
    assert isinstance(chrom, np.ndarray)

    new_chrom = chrom + np.random.randn(*chrom.shape) * 0.1
    mlp.set_chromosome(new_chrom)

    dummy_input = np.zeros(14, dtype=np.float32)
    output = mlp.forward(dummy_input)
    assert output.shape == (4,)
    assert np.all(output >= -1.0) and np.all(output <= 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ga_pipeline.py`
Expected: FAIL (`SimpleMLP` not defined)

- [ ] **Step 3: Implement `SimpleMLP`, GA training script, and `watch_match.py`**

- `SimpleMLP`: NumPy MLP with Tanh output activation to ensure strict `[-1.0, 1.0]` actions.
- GA Loop in `train_ga.py`:
  - Population size (e.g. 50 individuals).
  - Multi-round evaluations per generation to calculate average fitness (damage dealt + survival bonus - damage taken).
  - Top 10% elitism retention.
  - Gaussian mutation and uniform/blend crossover.
  - Checkpoint saving in `checkpoints/best_model_gen_N.npz`.
- `watch_match.py`:
  - Takes 2 model files (or `human`, `rule_based`, `random`).
  - Launches `env.render()` and steps at 60 FPS so user and friend can watch their models fight.

- [ ] **Step 4: Run test and run mini-training smoke test**

Run: `pytest tests/test_ga_pipeline.py`
Expected: PASS
Run: `python3 python/train_ga.py --generations 2 --population 10 --headless`
Expected: PASS with checkpoint generated

- [ ] **Step 5: Commit**

```bash
git add python/train_ga.py python/watch_match.py python/tank_env/neural_net.py tests/test_ga_pipeline.py
git commit -m "feat(ga): add genetic algorithm trainer, chromosome serialization, and match viewer"
```

---

### Task 7: Collaborator Handoff, Documentation & Multi-User Architecture Guide

**Files:**
- Create: `README.md`
- Create: `docs/COLLABORATION_GUIDE.md`
- Create: `docs/CUSTOM_INPUTS_GUIDE.md`
- Create: `examples/friend_template.py`

**Interfaces:**
- Consumes: Completed Playground codebase
- Produces: Clear setup instructions, architecture breakdown, and copy-paste template for the friend.

- [ ] **Step 1: Create collaborator template `examples/friend_template.py`**

Provide a clean template showing exactly where the friend implements:
1. Their custom input adapter (`class MyCustomObservationAdapter`).
2. Their neural network architecture of choice (PyTorch, TensorFlow, or custom NumPy).
3. How to export actions conforming to the `[-1.0, 1.0]` 4-float contract.
4. How to load and run against the user's saved champion in `watch_match.py`.

- [ ] **Step 2: Write documentation (`README.md`, `COLLABORATION_GUIDE.md`, `CUSTOM_INPUTS_GUIDE.md`)**

Document:
- Quickstart in 3 terminal commands (`python3 -m venv venv`, `pip install maturin pytest numpy`, `maturin develop --release`).
- How to test Rust engine standalone (`cargo test`).
- How to run the native Rust visualizer demo (`cargo run --release --bin standalone_viewer`).
- How to train models with GA (`python python/train_ga.py`).
- How to battle models (`python python/watch_match.py --tank1 my_model.npz --tank2 friend_model.npz`).
- Full observation dictionary schema breakdown.

- [ ] **Step 3: Verify the entire end-to-end workflow on Linux**

Run complete test suite:
- `cargo test --all`
- `pytest tests/`
- Run 1 generation GA check and verify model saving.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/ examples/
git commit -m "docs: add setup instructions, collaboration guide, and model sparring template"
```
