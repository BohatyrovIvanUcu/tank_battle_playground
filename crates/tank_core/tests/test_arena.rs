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

#[test]
fn test_obstacle_blocks_bullets_and_los() {
    let config = ArenaConfig::default();
    let mut arena = TankArena::new(config, 42);
    // Add partition wall in the center between x=380 and x=420, height 400
    arena.add_obstacle(380.0, 100.0, 40.0, 400.0);

    // Place tank 1 on left (x=200, y=300) facing right (0 rad)
    // Place tank 2 on right (x=600, y=300) facing left (PI rad)
    arena.set_positions_for_testing(200.0, 300.0, 0.0, 600.0, 300.0, std::f32::consts::PI);

    let t1_telem = arena.get_telemetry(1);
    // Wall is in the middle, so enemy should NOT be visible
    assert_eq!(t1_telem.enemy_visible, false);

    // Tank 1 shoots at Tank 2
    let shoot = TankAction { thrust: 0.0, turn: 0.0, turret_turn: 0.0, shoot: true };
    let noop = TankAction::default();
    arena.step(shoot, noop);
    assert_eq!(arena.bullets().len(), 1);

    // Bullet travels towards wall and hits it
    for _ in 0..50 {
        arena.step(noop, noop);
    }
    // Bullet must have been absorbed by the wall without hitting Tank 2
    let (_t1, t2) = arena.get_states();
    assert_eq!(t2.hp, 100.0);
    assert_eq!(arena.bullets().len(), 0);
}
