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

#[test]
fn test_asynchronous_parallel_action_updates() {
    let config = ArenaConfig::default();
    let mut arena = TankArena::new(config, 42);
    
    // Tank 1 responds quickly and sets action
    arena.set_tank_action(1, TankAction { thrust: 1.0, turn: 0.0, turret_turn: 0.0, shoot: false });
    // Tank 2 is still thinking (no action set or no-op)
    let _outcome = arena.tick();
    
    let (t1, t2) = arena.get_states();
    assert!(t1.speed > 0.0);
    assert_eq!(t2.speed, 0.0);
}
