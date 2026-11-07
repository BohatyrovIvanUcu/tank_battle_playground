pub mod math;
pub mod types;

#[cfg(feature = "gui")]
pub mod renderer;

// Re-exports
pub use math::Vec2;
pub use types::{
    ArenaConfig, BulletState, MatchOutcome, StepOutcome, TankAction, TankState, TankTelemetry,
};
