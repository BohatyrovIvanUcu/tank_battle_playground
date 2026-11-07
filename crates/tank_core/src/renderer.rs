use crate::arena::TankArena;
use crate::math::Vec2;
use crate::types::{BulletState, TankState};

#[cfg(feature = "gui")]
use minifb::{Key, Window, WindowOptions};

pub struct ArenaRenderer {
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<u32>,
    #[cfg(feature = "gui")]
    window: Option<Window>,
}

impl ArenaRenderer {
    pub fn new_headless(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buffer: vec![0; width * height],
            #[cfg(feature = "gui")]
            window: None,
        }
    }

    #[cfg(feature = "gui")]
    pub fn new_window(width: usize, height: usize, title: &str) -> Result<Self, String> {
        let mut opts = WindowOptions::default();
        opts.resize = false;
        opts.scale = minifb::Scale::X1;

        let mut window = Window::new(title, width, height, opts)
            .map_err(|e| format!("Failed to create window: {:?}", e))?;
        window.set_target_fps(60);

        Ok(Self {
            width,
            height,
            buffer: vec![0; width * height],
            window: Some(window),
        })
    }

    #[inline]
    pub fn buffer(&self) -> &[u32] {
        &self.buffer
    }

    pub fn is_open(&self) -> bool {
        #[cfg(feature = "gui")]
        if let Some(ref win) = self.window {
            return win.is_open() && !win.is_key_down(Key::Escape);
        }
        true
    }

    pub fn draw_arena(&mut self, arena: &TankArena) {
        // 1. Clear background (Dark modern terminal background #111418)
        self.clear(0x00111418);

        // 2. Draw subtle grid
        self.draw_grid(50, 0x001a1f26);

        // 3. Draw arena boundary walls
        self.draw_rect(2, 2, self.width - 4, self.height - 4, 0x00334155, false);

        // 3b. Draw Obstacles (Partition walls)
        for obs in arena.obstacles() {
            let x = (obs.x as usize).min(self.width);
            let y = (obs.y as usize).min(self.height);
            let w = (obs.width as usize).min(self.width.saturating_sub(x));
            let h = (obs.height as usize).min(self.height.saturating_sub(y));
            // Fill obstacle with solid slate
            self.draw_rect(x, y, w, h, 0x002A324B, true);
            // Border with highlight
            self.draw_rect(x, y, w, h, 0x0064748B, false);
        }

        // 4. Draw Bullets
        for bullet in arena.bullets() {
            if bullet.is_active {
                self.draw_bullet(bullet);
            }
        }

        // 5. Draw Tanks
        let (t1, t2) = arena.get_states();
        self.draw_tank(&t1, 0x0000E5FF, 0x0000B4D8); // Cyan for Tank 1
        self.draw_tank(&t2, 0x00FF3D00, 0x00D00000); // Orange/Red for Tank 2

        // 6. Draw HUD (Health bars and stats at the top)
        self.draw_hud(&t1, &t2, arena.step_count, arena.config.max_steps);
    }

    #[cfg(feature = "gui")]
    pub fn render_window(&mut self, arena: &TankArena) -> bool {
        self.draw_arena(arena);
        if let Some(ref mut win) = self.window {
            if win.is_open() && !win.is_key_down(Key::Escape) {
                let _ = win.update_with_buffer(&self.buffer, self.width, self.height);
                return true;
            }
        }
        false
    }

    // --- Drawing primitives ---

    #[inline]
    pub fn clear(&mut self, color: u32) {
        self.buffer.fill(color);
    }

    #[inline]
    pub fn set_pixel(&mut self, x: isize, y: isize, color: u32) {
        if x >= 0 && x < self.width as isize && y >= 0 && y < self.height as isize {
            self.buffer[y as usize * self.width + x as usize] = color;
        }
    }

    fn draw_grid(&mut self, step: usize, color: u32) {
        for x in (0..self.width).step_by(step) {
            for y in 0..self.height {
                self.set_pixel(x as isize, y as isize, color);
            }
        }
        for y in (0..self.height).step_by(step) {
            for x in 0..self.width {
                self.set_pixel(x as isize, y as isize, color);
            }
        }
    }

    fn draw_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: u32, fill: bool) {
        if fill {
            for dy in 0..h {
                for dx in 0..w {
                    self.set_pixel((x + dx) as isize, (y + dy) as isize, color);
                }
            }
        } else {
            for dx in 0..w {
                self.set_pixel((x + dx) as isize, y as isize, color);
                self.set_pixel((x + dx) as isize, (y + h.saturating_sub(1)) as isize, color);
            }
            for dy in 0..h {
                self.set_pixel(x as isize, (y + dy) as isize, color);
                self.set_pixel((x + w.saturating_sub(1)) as isize, (y + dy) as isize, color);
            }
        }
    }

    fn draw_circle(&mut self, cx: f32, cy: f32, radius: f32, color: u32, fill: bool) {
        let r = radius as isize;
        let r2 = (radius * radius) as isize;
        let icx = cx as isize;
        let icy = cy as isize;

        for dy in -r..=r {
            for dx in -r..=r {
                let dist_sq = dx * dx + dy * dy;
                if fill {
                    if dist_sq <= r2 {
                        self.set_pixel(icx + dx, icy + dy, color);
                    }
                } else if (dist_sq - r2).abs() <= r * 2 {
                    self.set_pixel(icx + dx, icy + dy, color);
                }
            }
        }
    }

    fn draw_line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: u32) {
        let mut x = x0;
        let mut y = y0;
        let dx = x1 - x0;
        let dy = y1 - y0;
        let steps = dx.abs().max(dy.abs()).ceil() as usize;

        if steps == 0 {
            self.set_pixel(x as isize, y as isize, color);
            return;
        }

        let x_inc = dx / steps as f32;
        let y_inc = dy / steps as f32;

        for _ in 0..=steps {
            self.set_pixel(x as isize, y as isize, color);
            x += x_inc;
            y += y_inc;
        }
    }

    fn draw_tank(&mut self, tank: &TankState, color_body: u32, color_turret: u32) {
        if !tank.is_alive {
            // Draw destroyed wreck (dim red X)
            self.draw_circle(tank.pos.x, tank.pos.y, 16.0, 0x004A1515, true);
            self.draw_line(
                tank.pos.x - 12.0,
                tank.pos.y - 12.0,
                tank.pos.x + 12.0,
                tank.pos.y + 12.0,
                0x00FF3333,
            );
            self.draw_line(
                tank.pos.x - 12.0,
                tank.pos.y + 12.0,
                tank.pos.x + 12.0,
                tank.pos.y - 12.0,
                0x00FF3333,
            );
            return;
        }

        // 1. Tank Hull (Circle + heading line)
        self.draw_circle(tank.pos.x, tank.pos.y, 16.0, color_body, true);
        self.draw_circle(tank.pos.x, tank.pos.y, 16.0, 0x00FFFFFF, false);

        // Body heading indicator
        let heading = Vec2::from_angle(tank.body_angle) * 16.0;
        self.draw_line(
            tank.pos.x,
            tank.pos.y,
            tank.pos.x + heading.x,
            tank.pos.y + heading.y,
            0x00000000,
        );

        // 2. Turret (Central cupola + long barrel)
        self.draw_circle(tank.pos.x, tank.pos.y, 7.0, color_turret, true);
        let barrel_end = tank.pos + Vec2::from_angle(tank.turret_angle) * 24.0;
        self.draw_line(tank.pos.x, tank.pos.y, barrel_end.x, barrel_end.y, 0x00FFFFFF);

        // 3. Health bar above tank
        let bar_w = 32;
        let bar_h = 4;
        let bar_x = (tank.pos.x - (bar_w as f32 * 0.5)) as usize;
        let bar_y = (tank.pos.y - 26.0).max(4.0) as usize;

        // Background (empty health)
        self.draw_rect(bar_x, bar_y, bar_w, bar_h, 0x00441111, true);

        // Foreground (current HP)
        let hp_ratio = (tank.hp / 100.0).clamp(0.0, 1.0);
        let cur_w = (bar_w as f32 * hp_ratio).round() as usize;
        let hp_color = if hp_ratio > 0.5 {
            0x0000FF66
        } else if hp_ratio > 0.25 {
            0x00FFAA00
        } else {
            0x00FF2222
        };
        self.draw_rect(bar_x, bar_y, cur_w, bar_h, hp_color, true);
    }

    fn draw_bullet(&mut self, bullet: &BulletState) {
        // Bullet glow
        let color = if bullet.owner_id == 1 {
            0x0000E5FF
        } else {
            0x00FF3D00
        };
        self.draw_circle(bullet.pos.x, bullet.pos.y, bullet.radius, 0x00FFF44F, true);
        // Short trailing line
        let trail = bullet.pos - bullet.vel * 1.5;
        self.draw_line(bullet.pos.x, bullet.pos.y, trail.x, trail.y, color);
    }

    fn draw_hud(&mut self, t1: &TankState, t2: &TankState, step: usize, max_steps: usize) {
        // Draw top HUD banner
        let hud_height = 28;
        self.draw_rect(0, 0, self.width, hud_height, 0x000B0E14, true);
        self.draw_line(0.0, hud_height as f32, self.width as f32, hud_height as f32, 0x00334155);

        // Tank 1 HP indicator (Left)
        let t1_hp_ratio = (t1.hp / 100.0).clamp(0.0, 1.0);
        let t1_bar_w = (150.0 * t1_hp_ratio) as usize;
        self.draw_rect(20, 8, 150, 12, 0x00331111, true);
        self.draw_rect(20, 8, t1_bar_w, 12, 0x0000E5FF, true);

        // Tank 2 HP indicator (Right)
        let t2_hp_ratio = (t2.hp / 100.0).clamp(0.0, 1.0);
        let t2_bar_w = (150.0 * t2_hp_ratio) as usize;
        let t2_x = self.width.saturating_sub(170);
        self.draw_rect(t2_x, 8, 150, 12, 0x00331111, true);
        self.draw_rect(t2_x + (150 - t2_bar_w), 8, t2_bar_w, 12, 0x00FF3D00, true);

        // Step progress bar (Center)
        let center_x = self.width / 2 - 50;
        let progress = (step as f32 / max_steps as f32).clamp(0.0, 1.0);
        self.draw_rect(center_x, 10, 100, 6, 0x001E293B, true);
        self.draw_rect(center_x, 10, (100.0 * progress) as usize, 6, 0x0094A3B8, true);
    }
}
