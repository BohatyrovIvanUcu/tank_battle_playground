"""
Champion Tank Agent: ivan_bot_67 (Gen 3000 Self-Play Co-Evolution Champion)
Drop this file along with ivan_bot_67.npz into any tournament referee.
"""
from pathlib import Path
from tank_env import MLPAgent, CombatObservationAdapter


def make_agent():
    model_path = Path(__file__).parent / "ivan_bot_67.npz"
    if not model_path.exists():
        model_path = Path(__file__).parent / "best_model.npz"
    return MLPAgent.load(model_path, CombatObservationAdapter())
