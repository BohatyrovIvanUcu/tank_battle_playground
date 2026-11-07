from typing import Optional, Tuple, Dict, Any, List
import tank_arena

class TankEnvironment:
    """
    Gym-like multi-agent 2D Tank Battle Environment.
    Wraps the Rust PyTankArena simulation engine.
    """
    def __init__(
        self,
        width: float = 800.0,
        height: float = 600.0,
        max_steps: int = 1500,
        seed: int = 42,
    ):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.seed = seed
        self._arena = tank_arena.PyTankArena(
            width=width,
            height=height,
            max_steps=max_steps,
            seed=seed,
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the arena and returns raw telemetry for both tanks."""
        return self._arena.reset(seed)

    def set_action(self, tank_id: int, action: List[float]):
        """Sets the action for a specific tank (1 or 2) without advancing the physical tick."""
        if not (1 <= tank_id <= 2):
            raise ValueError(f"tank_id must be 1 or 2, got {tank_id}")
        self._arena.set_action(tank_id, list(action))

    def tick(self) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], bool, Dict[str, Any]]:
        """Advances physical simulation by one tick using currently set actions."""
        return self._arena.tick()

    def step(
        self,
        action1: Optional[List[float]] = None,
        action2: Optional[List[float]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], bool, Dict[str, Any]]:
        """
        Advances the environment by one step.
        Supports both synchronous parallel stepping:
            step(action1, action2)
        and asynchronous stepping where one agent may still be thinking:
            step(action1=new_action, action2=None)
        """
        if action1 is not None and action2 is not None:
            return self._arena.step(list(action1), list(action2))
        else:
            if action1 is not None:
                self._arena.set_action(1, list(action1))
            if action2 is not None:
                self._arena.set_action(2, list(action2))
            return self._arena.tick()

    def add_obstacle(self, x: float, y: float, width: float, height: float):
        """Adds an impenetrable rectangular obstacle/wall that blocks tanks, bullets, and line-of-sight."""
        self._arena.add_obstacle(float(x), float(y), float(width), float(height))

    def clear_obstacles(self):
        """Removes all obstacles from the arena."""
        self._arena.clear_obstacles()

    def render(self) -> bool:
        """Renders current state in a native 60 FPS window. Returns False if user closed window."""
        return self._arena.render()

    def close(self):
        """Closes the rendering window and releases GUI resources."""
        self._arena.close()
