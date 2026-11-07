# 🏆 Agent: IVAN_BOT_67

Champion bot trained with Genetic Algorithm Self-Play Co-Evolution.
- **Generation**: 3000
- **Validation Score**: Undefeated Champion (154 points in 90-match Round-Robin vs Gen 1000, 2000, 2600)
- **Architecture**: SimpleMLP with `CombatObservationAdapter`
- **Weights**: `ivan_bot_67.npz` (or `best_model.npz`)

## How to Run Duel

```bash
# Duel against rule-based bot:
PYTHONPATH=python python3 python/referee.py submissions/ivan_bot_67/ivan_bot_67.py rule_based --seeds 10

# Duel with visualizer:
PYTHONPATH=python python3 python/referee.py submissions/ivan_bot_67/ivan_bot_67.py rule_based --seeds 2 --render
```
