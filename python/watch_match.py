#!/usr/bin/env python3
import argparse
import time
import os
import numpy as np

from tank_env import (
    TankEnvironment,
    VectorObservationAdapter,
    LidarObservationAdapter,
    RuleBasedBot,
    RandomBot,
    SimpleMLP,
)

def load_agent(agent_spec: str):
    """
    Loads agent by string spec:
    - path to .npz model file
    - 'rule_based'
    - 'random'
    """
    if agent_spec == "rule_based":
        return RuleBasedBot(), None
    elif agent_spec == "random":
        return RandomBot(), None
    elif os.path.exists(agent_spec):
        model = SimpleMLP.load(agent_spec)
        # Determine adapter based on input dimension
        if model.input_dim == 14:
            adapter = VectorObservationAdapter()
        elif model.input_dim == 12:
            adapter = LidarObservationAdapter(num_rays=8)
        else:
            adapter = VectorObservationAdapter()
        return model, adapter
    else:
        raise ValueError(f"Unknown agent spec or missing file: {agent_spec}")

def get_action(agent, adapter, raw_obs):
    if adapter is not None:
        x = adapter.extract(raw_obs)
        return agent.forward(x)
    else:
        return agent.act(raw_obs)

def main():
    parser = argparse.ArgumentParser(description="Watch 2D Tank Battle match with live renderer")
    parser.add_argument("--tank1", type=str, default="rule_based", help="Path to tank1 model .npz or 'rule_based'/'random'")
    parser.add_argument("--tank2", type=str, default="random", help="Path to tank2 model .npz or 'rule_based'/'random'")
    parser.add_argument("--matches", type=int, default=3, help="Number of matches to play")
    parser.add_argument("--delay", type=float, default=0.016, help="Tick sleep delay in seconds (~60 FPS = 0.016)")
    args = parser.parse_args()

    agent1, adapter1 = load_agent(args.tank1)
    agent2, adapter2 = load_agent(args.tank2)

    print("=========================================================")
    print(" 🎮 Starting Live Tank 2D Battle")
    print(f" - Tank 1 (Cyan):   {args.tank1}")
    print(f" - Tank 2 (Orange): {args.tank2}")
    print(" - Press ESC in the game window to stop.")
    print("=========================================================")

    env = TankEnvironment(max_steps=1200)

    for match in range(1, args.matches + 1):
        print(f"\n--- Match {match}/{args.matches} ---")
        obs1_raw, obs2_raw = env.reset(seed=match * 37)
        done = False
        step = 0

        while not done:
            step += 1
            act1 = get_action(agent1, adapter1, obs1_raw)
            act2 = get_action(agent2, adapter2, obs2_raw)

            obs1_raw, obs2_raw, rewards, done, info = env.step(act1, act2)

            is_open = env.render()
            if not is_open:
                print("Window closed by user. Exiting.")
                return

            if args.delay > 0:
                time.sleep(args.delay)

        outcome = info.get("outcome", "unknown")
        print(f"Match {match} ended in {step} ticks! Outcome: {outcome}")
        print(f"Tank 1 HP: {obs1_raw['hp']:.1f} | Tank 2 HP: {obs2_raw['hp']:.1f}")
        time.sleep(1.0)

    env.close()
    print("\n🏁 All matches finished.")

if __name__ == "__main__":
    main()
