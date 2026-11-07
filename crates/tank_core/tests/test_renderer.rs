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
