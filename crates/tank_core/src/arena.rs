use crate::math::{
    clamp_action, circle_rect_resolve, normalize_angle, ray_bounds_intersect, ray_circle_intersect,
    ray_rect_intersect, segment_rect_intersect, Vec2,
};
use crate::types::{
    ArenaConfig, BulletState, MatchOutcome, Obstacle, StepOutcome, TankAction, TankState,
    TankTelemetry,
};

#[derive(Debug, Clone)]
pub struct TankArena {
    pub config: ArenaConfig,
    pub step_count: usize,
    pub tank1: TankState,
    pub tank2: TankState,
    pub action1: TankAction,
    pub action2: TankAction,
    pub bullets: Vec<BulletState>,
    pub obstacles: Vec<Obstacle>,
    pub next_bullet_id: u32,
    pub done: bool,
    pub outcome: MatchOutcome,
}

impl TankArena {
    pub fn new(config: ArenaConfig, _seed: u64) -> Self {
        let mut arena = Self {
            config,
            step_count: 0,
            tank1: TankState::new(1, Vec2::new(100.0, config.height * 0.5), 0.0),
            tank2: TankState::new(
                2,
                Vec2::new(config.width - 100.0, config.height * 0.5),
                std::f32::consts::PI,
            ),
            action1: TankAction::default(),
            action2: TankAction::default(),
            bullets: Vec::with_capacity(32),
            obstacles: Vec::new(),
            next_bullet_id: 1,
            done: false,
            outcome: MatchOutcome::InProgress,
        };
        arena.reset_tanks();
        arena
    }

    pub fn reset_tanks(&mut self) {
        self.step_count = 0;
        self.bullets.clear();
        self.done = false;
        self.outcome = MatchOutcome::InProgress;

        self.tank1 = TankState::new(
            1,
            Vec2::new(self.config.width * 0.15, self.config.height * 0.5),
            0.0,
        );
        self.tank2 = TankState::new(
            2,
            Vec2::new(self.config.width * 0.85, self.config.height * 0.5),
            std::f32::consts::PI,
        );
        self.action1 = TankAction::default();
        self.action2 = TankAction::default();
    }

    pub fn add_obstacle(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.obstacles.push(Obstacle::new(x, y, width, height));
    }

    pub fn clear_obstacles(&mut self) {
        self.obstacles.clear();
    }

    #[inline]
    pub fn obstacles(&self) -> &[Obstacle] {
        &self.obstacles
    }

    #[inline]
    pub fn get_states(&self) -> (TankState, TankState) {
        (self.tank1, self.tank2)
    }

    #[inline]
    pub fn bullets(&self) -> &[BulletState] {
        &self.bullets
    }

    pub fn set_positions_for_testing(
        &mut self,
        x1: f32,
        y1: f32,
        angle1: f32,
        x2: f32,
        y2: f32,
        angle2: f32,
    ) {
        self.tank1.pos = Vec2::new(x1, y1);
        self.tank1.body_angle = angle1;
        self.tank1.turret_angle = angle1;

        self.tank2.pos = Vec2::new(x2, y2);
        self.tank2.body_angle = angle2;
        self.tank2.turret_angle = angle2;
    }

    /// Set an action for a specific tank independently (supports asynchronous / parallel thinking agents)
    pub fn set_tank_action(&mut self, tank_id: u8, action: TankAction) {
        if tank_id == 1 {
            self.action1 = action;
        } else if tank_id == 2 {
            self.action2 = action;
        }
    }

    /// Advance the simulation by one physical tick using the currently active actions
    pub fn tick(&mut self) -> StepOutcome {
        if self.done {
            return StepOutcome {
                step_count: self.step_count,
                done: true,
                outcome: self.outcome,
                reward1: 0.0,
                reward2: 0.0,
            };
        }

        self.step_count += 1;
        let mut hit_dealt1 = 0;
        let mut hit_dealt2 = 0;

        // 1. Decrement reload timers
        if self.tank1.reload_cooldown > 0 {
            self.tank1.reload_cooldown -= 1;
        }
        if self.tank2.reload_cooldown > 0 {
            self.tank2.reload_cooldown -= 1;
        }

        // 2. Process Actions for Tank 1
        Self::apply_tank_movement(&mut self.tank1, self.action1, &self.config);
        // Process Actions for Tank 2
        Self::apply_tank_movement(&mut self.tank2, self.action2, &self.config);

        // 3. Process Shoots
        if self.action1.shoot && self.tank1.reload_cooldown == 0 && self.tank1.is_alive {
            self.spawn_bullet(1);
            self.tank1.reload_cooldown = self.config.reload_ticks;
            self.tank1.shots_fired += 1;
        }
        if self.action2.shoot && self.tank2.reload_cooldown == 0 && self.tank2.is_alive {
            self.spawn_bullet(2);
            self.tank2.reload_cooldown = self.config.reload_ticks;
            self.tank2.shots_fired += 1;
        }

        // 4. Resolve Tank vs Tank collisions
        let delta = self.tank2.pos - self.tank1.pos;
        let dist = delta.length();
        let min_dist = self.config.tank_radius * 2.0;
        if dist < min_dist && dist > 1e-5 {
            let overlap = min_dist - dist;
            let normal = delta.normalize();
            // Separate them symmetrically
            self.tank1.pos -= normal * (overlap * 0.5);
            self.tank2.pos += normal * (overlap * 0.5);
        }

        // 4b. Resolve Tank vs Obstacle collisions
        for obs in &self.obstacles {
            circle_rect_resolve(
                &mut self.tank1.pos,
                self.config.tank_radius,
                obs.x,
                obs.y,
                obs.width,
                obs.height,
            );
            circle_rect_resolve(
                &mut self.tank2.pos,
                self.config.tank_radius,
                obs.x,
                obs.y,
                obs.width,
                obs.height,
            );
        }

        // 5. Clamp positions inside arena boundaries
        Self::clamp_to_bounds(&mut self.tank1, &self.config);
        Self::clamp_to_bounds(&mut self.tank2, &self.config);

        // 6. Update Bullets & check collisions against both tanks, obstacles, and walls
        let mut active_bullets = Vec::with_capacity(self.bullets.len());
        for mut b in self.bullets.drain(..) {
            if !b.is_active {
                continue;
            }

            let next_pos = b.pos + b.vel;
            let ray_dir = b.vel.normalize();
            let travel_dist = b.vel.length();

            // Check collision with obstacles (partition walls)
            for obs in &self.obstacles {
                if segment_rect_intersect(b.pos, next_pos, obs.x, obs.y, obs.width, obs.height) {
                    b.is_active = false;
                    break;
                }
            }

            if !b.is_active {
                continue;
            }

            // Check collision with Tank 1
            if b.owner_id != 1 && self.tank1.is_alive {
                if let Some((hit_dist, _)) =
                    ray_circle_intersect(b.pos, ray_dir, self.tank1.pos, self.config.tank_radius)
                {
                    if hit_dist <= travel_dist {
                        // Impact on Tank 1
                        self.tank1.hp = (self.tank1.hp - self.config.bullet_damage).max(0.0);
                        self.tank1.hits_taken += 1;
                        self.tank2.hits_dealt += 1;
                        hit_dealt2 += 1;
                        b.is_active = false;
                        if self.tank1.hp <= 0.0 {
                            self.tank1.is_alive = false;
                        }
                    }
                }
            }

            // Check collision with Tank 2
            if b.is_active && b.owner_id != 2 && self.tank2.is_alive {
                if let Some((hit_dist, _)) =
                    ray_circle_intersect(b.pos, ray_dir, self.tank2.pos, self.config.tank_radius)
                {
                    if hit_dist <= travel_dist {
                        // Impact on Tank 2
                        self.tank2.hp = (self.tank2.hp - self.config.bullet_damage).max(0.0);
                        self.tank2.hits_taken += 1;
                        self.tank1.hits_dealt += 1;
                        hit_dealt1 += 1;
                        b.is_active = false;
                        if self.tank2.hp <= 0.0 {
                            self.tank2.is_alive = false;
                        }
                    }
                }
            }

            // Check collision with arena boundaries
            if b.is_active {
                if next_pos.x <= 0.0
                    || next_pos.x >= self.config.width
                    || next_pos.y <= 0.0
                    || next_pos.y >= self.config.height
                {
                    b.is_active = false;
                }
            }

            if b.is_active {
                b.pos = next_pos;
                active_bullets.push(b);
            }
        }
        self.bullets = active_bullets;

        // 7. Check Win/Draw conditions
        if !self.tank1.is_alive && !self.tank2.is_alive {
            self.done = true;
            self.outcome = MatchOutcome::DrawTie;
        } else if !self.tank1.is_alive {
            self.done = true;
            self.outcome = MatchOutcome::Tank2Win;
        } else if !self.tank2.is_alive {
            self.done = true;
            self.outcome = MatchOutcome::Tank1Win;
        } else if self.step_count >= self.config.max_steps {
            self.done = true;
            if self.tank1.hp > self.tank2.hp {
                self.outcome = MatchOutcome::Tank1Win;
            } else if self.tank2.hp > self.tank1.hp {
                self.outcome = MatchOutcome::Tank2Win;
            } else {
                self.outcome = MatchOutcome::DrawTimeout;
            }
        }

        // 8. Compute reward signals for ML/GA fitness
        let mut r1 = 0.01; // living bonus
        let mut r2 = 0.01;

        r1 += hit_dealt1 as f32 * 15.0 - hit_dealt2 as f32 * 10.0;
        r2 += hit_dealt2 as f32 * 15.0 - hit_dealt1 as f32 * 10.0;

        if self.done {
            match self.outcome {
                MatchOutcome::Tank1Win => {
                    r1 += 100.0;
                    r2 -= 100.0;
                }
                MatchOutcome::Tank2Win => {
                    r2 += 100.0;
                    r1 -= 100.0;
                }
                MatchOutcome::DrawTie | MatchOutcome::DrawTimeout => {
                    // Slight penalty for draw
                    r1 -= 10.0;
                    r2 -= 10.0;
                }
                _ => {}
            }
        }

        StepOutcome {
            step_count: self.step_count,
            done: self.done,
            outcome: self.outcome,
            reward1: r1,
            reward2: r2,
        }
    }

    /// Convenience step method setting both actions at once and advancing one tick
    pub fn step(&mut self, action1: TankAction, action2: TankAction) -> StepOutcome {
        self.action1 = action1;
        self.action2 = action2;
        self.tick()
    }

    fn apply_tank_movement(tank: &mut TankState, action: TankAction, config: &ArenaConfig) {
        if !tank.is_alive {
            tank.speed = 0.0;
            tank.vel = Vec2::ZERO;
            return;
        }

        let thrust = clamp_action(action.thrust);
        let turn = clamp_action(action.turn);
        let turret_turn = clamp_action(action.turret_turn);

        // Turn body
        tank.body_angle = normalize_angle(tank.body_angle + turn * config.tank_turn_speed);
        // Turn turret (relative to world angle, steered by turret_turn)
        tank.turret_angle = normalize_angle(tank.turret_angle + turret_turn * config.turret_turn_speed);

        // Accelerate / decelerate
        let target_speed = if thrust >= 0.0 {
            thrust * config.tank_max_speed
        } else {
            thrust * config.tank_max_reverse_speed
        };

        tank.speed += (target_speed - tank.speed) * config.tank_accel;
        let heading = Vec2::from_angle(tank.body_angle);
        tank.vel = heading * tank.speed;
        tank.pos += tank.vel;
    }

    fn clamp_to_bounds(tank: &mut TankState, config: &ArenaConfig) {
        let r = config.tank_radius;
        if tank.pos.x < r {
            tank.pos.x = r;
            if tank.vel.x < 0.0 {
                tank.vel.x = 0.0;
            }
        } else if tank.pos.x > config.width - r {
            tank.pos.x = config.width - r;
            if tank.vel.x > 0.0 {
                tank.vel.x = 0.0;
            }
        }

        if tank.pos.y < r {
            tank.pos.y = r;
            if tank.vel.y < 0.0 {
                tank.vel.y = 0.0;
            }
        } else if tank.pos.y > config.height - r {
            tank.pos.y = config.height - r;
            if tank.vel.y > 0.0 {
                tank.vel.y = 0.0;
            }
        }
    }

    fn spawn_bullet(&mut self, owner_id: u8) {
        let tank = if owner_id == 1 { &self.tank1 } else { &self.tank2 };
        let muzzle_dir = Vec2::from_angle(tank.turret_angle);
        let spawn_pos = tank.pos + muzzle_dir * (self.config.tank_radius + 5.0);
        let bullet_vel = muzzle_dir * self.config.bullet_speed;

        self.bullets.push(BulletState {
            id: self.next_bullet_id,
            owner_id,
            pos: spawn_pos,
            vel: bullet_vel,
            radius: self.config.bullet_radius,
            is_active: true,
        });
        self.next_bullet_id += 1;
    }

    /// Extract telemetry for a given tank id (1 or 2)
    pub fn get_telemetry(&self, tank_id: u8) -> TankTelemetry {
        let (me, enemy) = if tank_id == 1 {
            (&self.tank1, &self.tank2)
        } else {
            (&self.tank2, &self.tank1)
        };

        let enemy_rel = enemy.pos - me.pos;
        let enemy_dist = enemy_rel.length();
        let angle_to_enemy = normalize_angle(enemy_rel.to_angle() - me.body_angle);

        // 8 Lidar rays: angles at -135, -90, -45, 0, 45, 90, 135, 180 degrees relative to body
        let mut lidar_rays = [1.0f32; 8];
        let ray_angles = [
            -0.75 * std::f32::consts::PI,
            -0.50 * std::f32::consts::PI,
            -0.25 * std::f32::consts::PI,
            0.0,
            0.25 * std::f32::consts::PI,
            0.50 * std::f32::consts::PI,
            0.75 * std::f32::consts::PI,
            std::f32::consts::PI,
        ];

        let max_lidar_range = (self.config.width * self.config.width
            + self.config.height * self.config.height)
            .sqrt();

        for (i, &rel_a) in ray_angles.iter().enumerate() {
            let world_angle = normalize_angle(me.body_angle + rel_a);
            let dir = Vec2::from_angle(world_angle);

            let mut min_t = max_lidar_range;
            // Check wall intersection
            if let Some(t_wall) =
                ray_bounds_intersect(me.pos, dir, self.config.width, self.config.height)
            {
                if t_wall < min_t {
                    min_t = t_wall;
                }
            }

            // Check enemy tank intersection
            if enemy.is_alive {
                if let Some((t_enemy, _)) =
                    ray_circle_intersect(me.pos, dir, enemy.pos, self.config.tank_radius)
                {
                    if t_enemy < min_t {
                        min_t = t_enemy;
                    }
                }
            }

            // Check obstacle intersections
            for obs in &self.obstacles {
                if let Some(t_obs) =
                    ray_rect_intersect(me.pos, dir, obs.x, obs.y, obs.width, obs.height)
                {
                    if t_obs < min_t {
                        min_t = t_obs;
                    }
                }
            }

            lidar_rays[i] = (min_t / max_lidar_range).clamp(0.0, 1.0);
        }

        // Line of Sight (LOS): is the enemy tank visible or blocked by an obstacle?
        let enemy_visible = enemy.is_alive
            && !self.obstacles.iter().any(|obs| {
                segment_rect_intersect(me.pos, enemy.pos, obs.x, obs.y, obs.width, obs.height)
            });

        // Bullets telemetry relative to this tank
        let mut bullets_info = Vec::with_capacity(self.bullets.len());
        for b in &self.bullets {
            let rel = b.pos - me.pos;
            let is_enemy = if b.owner_id != me.id { 1.0 } else { 0.0 };
            bullets_info.push([
                rel.x / self.config.width,
                rel.y / self.config.height,
                b.vel.x / self.config.bullet_speed,
                b.vel.y / self.config.bullet_speed,
                is_enemy,
            ]);
        }

        TankTelemetry {
            x: me.pos.x / self.config.width,
            y: me.pos.y / self.config.height,
            vx: me.vel.x / self.config.tank_max_speed,
            vy: me.vel.y / self.config.tank_max_speed,
            speed: me.speed / self.config.tank_max_speed,
            body_angle: me.body_angle,
            turret_angle: me.turret_angle,
            hp: me.hp,
            reload_cooldown: me.reload_cooldown as f32 / self.config.reload_ticks as f32,
            enemy_rel_x: enemy_rel.x / self.config.width,
            enemy_rel_y: enemy_rel.y / self.config.height,
            enemy_dist: enemy_dist / max_lidar_range,
            enemy_angle_to: angle_to_enemy,
            enemy_turret_angle: enemy.turret_angle,
            enemy_hp: enemy.hp,
            enemy_visible,
            lidar_rays,
            bullets: bullets_info,
        }
    }
}
