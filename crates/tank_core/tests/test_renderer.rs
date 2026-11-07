use tank_core::arena::TankArena;
use tank_core::renderer::{ArenaRenderer, HUD_HEIGHT};
use tank_core::types::ArenaConfig;

#[test]
fn test_renderer_buffer_generation() {
    let arena = TankArena::new(ArenaConfig::default(), 42);
    let cfg = arena.config;
    let (w, h) = (cfg.width as usize, cfg.height as usize);
    let mut renderer = ArenaRenderer::new_headless(w, h);
    renderer.draw_arena(&arena);
    let buf = renderer.buffer();
    // the control strip sits above the field, it is not part of the arena
    assert_eq!(buf.len(), w * (h + HUD_HEIGHT));
    // the HUD strip is fully painted (nothing of the field leaks into it)
    assert!(buf[..w * HUD_HEIGHT].iter().all(|&p| p != 0));
    // field row 0 starts right below the strip and shows the arena, not the HUD colour
    assert_ne!(buf[w * HUD_HEIGHT + w / 2], 0x000B0E14);
    // Ensure screen is not completely blank (black)
    assert!(buf.iter().any(|&p| p != 0x000000));
}
