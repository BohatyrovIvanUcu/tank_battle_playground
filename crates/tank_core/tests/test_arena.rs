use tank_core::arena::TankArena;
use tank_core::types::{ArenaConfig, TankAction};

const NOOP: TankAction = TankAction { thrust: 0.0, turn: 0.0, turret_turn: 0.0, shoot: false };
const SHOOT: TankAction = TankAction { thrust: 0.0, turn: 0.0, turret_turn: 0.0, shoot: true };

/// Arena with a known-empty map, for physics tests independent of the random layout.
fn empty_arena() -> TankArena {
    let mut arena = TankArena::new(ArenaConfig::default(), 42);
    arena.clear_obstacles();
    arena
}

#[test]
fn test_arena_initialization_and_step() {
    let mut arena = TankArena::new(ArenaConfig::default(), 42);
    let (t1, t2) = arena.get_states();
    assert_eq!(t1.hp, 100.0);
    assert_eq!(t2.hp, 100.0);

    let action = TankAction { thrust: 1.0, ..NOOP };
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
    arena.clear_obstacles();
    arena.set_positions_for_testing(100.0, 100.0, 0.0, 200.0, 100.0, std::f32::consts::PI);
    arena.step(SHOOT, NOOP);
    assert_eq!(arena.bullets().len(), 1);

    for _ in 0..30 {
        arena.step(NOOP, NOOP);
    }
    let (_t1, t2) = arena.get_states();
    assert_eq!(t2.hp, 75.0);
}

#[test]
fn test_asynchronous_parallel_action_updates() {
    let mut arena = empty_arena();
    arena.set_tank_action(1, TankAction { thrust: 1.0, ..NOOP });
    let _outcome = arena.tick();

    let (t1, t2) = arena.get_states();
    assert!(t1.speed > 0.0);
    assert_eq!(t2.speed, 0.0);
}

#[test]
fn test_obstacle_blocks_bullets_and_los() {
    let mut arena = empty_arena();
    arena.add_obstacle(380.0, 100.0, 40.0, 400.0);
    arena.set_positions_for_testing(200.0, 300.0, 0.0, 600.0, 300.0, std::f32::consts::PI);

    assert!(!arena.get_telemetry(1).enemy_visible);

    arena.step(SHOOT, NOOP);
    assert_eq!(arena.bullets().len(), 1);
    for _ in 0..50 {
        arena.step(NOOP, NOOP);
    }
    let (_t1, t2) = arena.get_states();
    assert_eq!(t2.hp, 100.0);
    assert_eq!(arena.bullets().len(), 0);
}

#[test]
fn test_same_seed_same_map_different_seed_different_map() {
    let a = TankArena::new(ArenaConfig::default(), 7);
    let b = TankArena::new(ArenaConfig::default(), 7);
    assert_eq!(a.obstacles(), b.obstacles());
    assert_eq!(a.get_states(), b.get_states());

    let differs = (100..120u64).any(|s| {
        let c = TankArena::new(ArenaConfig::default(), s);
        c.obstacles() != a.obstacles() || c.get_states() != a.get_states()
    });
    assert!(differs, "different seeds must produce different episodes");

    // reset(seed) on an existing arena reproduces a fresh one
    let mut r = TankArena::new(ArenaConfig::default(), 1);
    r.reset(7);
    assert_eq!(r.obstacles(), a.obstacles());
    assert_eq!(r.get_states(), a.get_states());
}

#[test]
fn test_maps_are_symmetric_fair_and_passable() {
    let cfg = ArenaConfig::default();
    let clearance = 3.0 * cfg.tank_radius;
    let mut with_obstacles = 0;
    for seed in 0..300u64 {
        let arena = TankArena::new(cfg, seed);
        let (t1, t2) = arena.get_states();
        let obs = arena.obstacles();
        if !obs.is_empty() {
            with_obstacles += 1;
        }

        // Spawns: point mirror of each other, opposite headings, inside the arena.
        assert!((t1.pos.x + t2.pos.x - cfg.width).abs() < 1e-3, "seed {seed}");
        assert!((t1.pos.y + t2.pos.y - cfg.height).abs() < 1e-3, "seed {seed}");
        assert!(t1.pos.x < cfg.width * 0.5);

        // Every obstacle has its mirrored twin.
        for o in obs {
            let m = o.mirrored(cfg.width, cfg.height);
            assert!(
                obs.iter().any(|e| (e.x - m.x).abs() < 1e-3
                    && (e.y - m.y).abs() < 1e-3
                    && (e.width - m.width).abs() < 1e-3
                    && (e.height - m.height).abs() < 1e-3),
                "seed {seed}: obstacle {o:?} has no twin"
            );
            // Gap to border and to spawns (so tanks fit through every passage).
            assert!(o.x >= clearance && o.y >= clearance, "seed {seed}");
            assert!(o.x + o.width <= cfg.width - clearance, "seed {seed}");
            assert!(o.y + o.height <= cfg.height - clearance, "seed {seed}");
            assert!(!o.near(t1.pos, clearance) && !o.near(t2.pos, clearance), "seed {seed}");
        }
        // Gap between distinct obstacles.
        for (i, a) in obs.iter().enumerate() {
            for b in obs.iter().skip(i + 1) {
                assert!(!a.overlaps(b, clearance - 1e-3), "seed {seed}: {a:?} vs {b:?}");
            }
        }
    }
    assert!(with_obstacles > 250, "maps should almost always contain obstacles");
}

#[test]
fn test_fog_of_war_reports_last_known_enemy() {
    let mut arena = empty_arena();
    arena.set_positions_for_testing(200.0, 300.0, 0.0, 600.0, 300.0, std::f32::consts::PI);
    let seen = arena.get_telemetry(1);
    assert!(seen.enemy_visible);
    assert_eq!(seen.enemy_last_seen, 0.0);

    // Wall appears; tank 2 drives away (up) while hidden.
    arena.add_obstacle(380.0, 100.0, 40.0, 400.0);
    for _ in 0..20 {
        arena.step(NOOP, TankAction { thrust: 1.0, turn: 1.0, ..NOOP });
    }
    let (t1, t2) = arena.get_states();
    assert!((t2.pos.x - 600.0).abs() > 1.0 || (t2.pos.y - 300.0).abs() > 1.0, "enemy moved");

    let hidden = arena.get_telemetry(1);
    assert!(!hidden.enemy_visible);
    assert!(hidden.enemy_last_seen > 0.0);
    // Still reports the position from when it was last seen (600, 300), not the live one.
    let expect_rel_x = (600.0 - t1.pos.x) / arena.config.width;
    let expect_rel_y = (300.0 - t1.pos.y) / arena.config.height;
    assert!((hidden.enemy_rel_x - expect_rel_x).abs() < 1e-4);
    assert!((hidden.enemy_rel_y - expect_rel_y).abs() < 1e-4);
    assert_eq!(hidden.enemy_hp, 100.0);

    // The map is public in the telemetry.
    assert!(hidden.obstacles.iter().any(|o| (o[0] - 380.0 / 800.0).abs() < 1e-4));
}

#[test]
fn test_shot_against_wall_does_not_pass_through() {
    let mut arena = empty_arena();
    // Thin wall directly in front of tank 1's muzzle; tank 1 pressed against it.
    arena.add_obstacle(120.0, 250.0, 2.0, 100.0);
    arena.set_positions_for_testing(100.0, 300.0, 0.0, 300.0, 300.0, std::f32::consts::PI);
    arena.step(SHOOT, NOOP);
    assert_eq!(arena.bullets().len(), 0, "muzzle blocked: no bullet");
    for _ in 0..60 {
        arena.step(NOOP, NOOP);
    }
    assert_eq!(arena.get_states().1.hp, 100.0);
}

#[test]
fn test_reset_clears_episode_state() {
    let mut arena = TankArena::new(ArenaConfig::default(), 3);
    arena.step(SHOOT, SHOOT);
    arena.reset(3);
    assert_eq!(arena.step_count, 0);
    assert!(arena.bullets().is_empty());
    let t = arena.get_telemetry(1);
    assert_eq!(t.enemy_last_seen, 0.0);
}
