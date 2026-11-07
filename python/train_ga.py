#!/usr/bin/env python3
import argparse
import os
import random
import time
import numpy as np

from tank_env import (
    TankEnvironment,
    VectorObservationAdapter,
    LidarObservationAdapter,
    RuleBasedBot,
    RandomBot,
    SimpleMLP,
)

class GAPopulation:
    def __init__(
        self,
        pop_size: int,
        input_dim: int,
        hidden_dims: list,
        output_dim: int = 4,
        mutation_rate: float = 0.10,
        mutation_power: float = 0.20,
        elitism_ratio: float = 0.15,
    ):
        self.pop_size = pop_size
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.mutation_rate = mutation_rate
        self.mutation_power = mutation_power
        self.elitism_ratio = elitism_ratio

        self.population: list[SimpleMLP] = [
            SimpleMLP(input_dim, hidden_dims, output_dim) for _ in range(pop_size)
        ]

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        mutated = chromosome.copy()
        mask = np.random.rand(len(mutated)) < self.mutation_rate
        noise = np.random.randn(len(mutated)) * self.mutation_power
        mutated += mask * noise
        return mutated.astype(np.float32)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        # Uniform blend crossover
        alpha = np.random.uniform(0.0, 1.0, size=len(parent1)).astype(np.float32)
        child = alpha * parent1 + (1.0 - alpha) * parent2
        return child

    def evolve(self, fitnesses: list[float]) -> tuple[SimpleMLP, float, float]:
        sorted_indices = np.argsort(fitnesses)[::-1]
        best_idx = sorted_indices[0]
        best_individual = self.population[best_idx]
        best_fitness = fitnesses[best_idx]
        mean_fitness = float(np.mean(fitnesses))

        num_elites = max(1, int(self.pop_size * self.elitism_ratio))
        new_population = []

        # 1. Elitism
        for i in range(num_elites):
            elite_model = SimpleMLP(self.input_dim, self.hidden_dims, self.output_dim)
            elite_model.set_chromosome(self.population[sorted_indices[i]].get_chromosome().copy())
            new_population.append(elite_model)

        # 2. Tournament selection & reproduction
        tournament_k = 3
        while len(new_population) < self.pop_size:
            # Pick parents via tournament
            t1 = random.sample(range(self.pop_size), tournament_k)
            t2 = random.sample(range(self.pop_size), tournament_k)
            p1_idx = max(t1, key=lambda idx: fitnesses[idx])
            p2_idx = max(t2, key=lambda idx: fitnesses[idx])

            c1 = self.population[p1_idx].get_chromosome()
            c2 = self.population[p2_idx].get_chromosome()

            child_chrom = self.crossover(c1, c2)
            child_chrom = self.mutate(child_chrom)

            child_model = SimpleMLP(self.input_dim, self.hidden_dims, self.output_dim)
            child_model.set_chromosome(child_chrom)
            new_population.append(child_model)

        self.population = new_population
        return best_individual, best_fitness, mean_fitness


def evaluate_individual_vs_bot(
    model: SimpleMLP,
    adapter,
    bot,
    env: TankEnvironment,
    rounds: int = 2,
) -> float:
    total_score = 0.0
    for r in range(rounds):
        obs1_raw, obs2_raw = env.reset(seed=r * 100 + 42)
        done = False
        step = 0
        while not done:
            step += 1
            x1 = adapter.extract(obs1_raw)
            a1 = model.forward(x1)
            a2 = bot.act(obs2_raw)

            obs1_raw, obs2_raw, rewards, done, info = env.step(a1, a2)
            total_score += rewards[0]

    return total_score / rounds


def main():
    parser = argparse.ArgumentParser(description="Train Tank Battle Agent using Genetic Algorithm")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--population", type=int, default=20, help="Population size")
    parser.add_argument("--rounds", type=int, default=2, help="Rounds per evaluation")
    parser.add_argument("--adapter", type=str, choices=["vector", "lidar"], default="vector")
    parser.add_argument("--opponent", type=str, choices=["rule_based", "random"], default="rule_based")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    adapter = VectorObservationAdapter() if args.adapter == "vector" else LidarObservationAdapter()
    bot = RuleBasedBot() if args.opponent == "rule_based" else RandomBot()

    print("=========================================================")
    print(" 🚀 Starting Genetic Algorithm Training for 2D Tanks")
    print(f" - Generations: {args.generations}")
    print(f" - Population:  {args.population}")
    print(f" - Adapter:     {args.adapter} (dim={adapter.output_dim})")
    print(f" - Opponent:    {args.opponent}")
    print(f" - Save dir:    {args.save_dir}")
    print("=========================================================")

    env = TankEnvironment(max_steps=800)
    ga = GAPopulation(
        pop_size=args.population,
        input_dim=adapter.output_dim,
        hidden_dims=[24, 24],
        output_dim=4,
    )

    best_overall_fitness = -float("inf")
    best_overall_model: SimpleMLP = ga.population[0]

    start_time = time.time()
    for gen in range(1, args.generations + 1):
        fitnesses = []
        for ind in ga.population:
            score = evaluate_individual_vs_bot(ind, adapter, bot, env, rounds=args.rounds)
            fitnesses.append(score)

        best_ind, best_fit, mean_fit = ga.evolve(fitnesses)
        print(f"Gen {gen:03d}/{args.generations:03d} | Best Fitness: {best_fit:7.2f} | Mean Fitness: {mean_fit:7.2f}")

        if best_fit > best_overall_fitness:
            best_overall_fitness = best_fit
            best_overall_model = best_ind
            checkpoint_path = os.path.join(args.save_dir, "best_model.npz")
            best_overall_model.save(checkpoint_path)

        if gen % 5 == 0 or gen == args.generations:
            gen_path = os.path.join(args.save_dir, f"model_gen_{gen}.npz")
            best_ind.save(gen_path)

    elapsed = time.time() - start_time
    print("=========================================================")
    print(f" ✨ Training complete in {elapsed:.2f}s!")
    print(f" 🏆 Best overall fitness: {best_overall_fitness:.2f}")
    print(f" 💾 Best model saved to: {os.path.join(args.save_dir, 'best_model.npz')}")
    print("=========================================================")

if __name__ == "__main__":
    main()
