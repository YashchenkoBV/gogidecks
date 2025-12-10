# genalg_benchmark.py
"""
Benchmark runner for genalg_main.py

Runs experiments varying:
- mutation rate
- population size
- tournament k (selection pressure)

and plots best fitness over generations for each setting.
"""

import argparse
import random
from typing import Sequence, Tuple, List

import matplotlib.pyplot as plt

import genalg_main as ga


# ------------------------------------------------------------
# Small GA loop for benchmarking WITH history tracking
# ------------------------------------------------------------

def run_experiment(
    pop_size: int,
    generations: int,
    tournament_k: int,
    crossover_rate: float,
    mutation_rate_gene: float,
    same_class_prob: float,
    elitism: int,
    forced_slugs: Sequence[str],
    seed: int | None = None,
) -> Tuple[List[float], List[float]]:
    """
    Run a GA using operators from genalg_main, and return:
    - best_fitness_history: best fitness per generation (len = generations + 1)
    - avg_fitness_history: average fitness per generation (len = generations + 1)
    """

    if seed is not None:
        random.seed(seed)

    forced_indices = ga.slugs_to_indices(forced_slugs)

    # ---- Initial population (may be smaller than requested if constraints are tight) ----
    population = ga.initial_population(pop_size, forced_slugs)
    pop_size = len(population)

    fitnesses = [ga.chromosome_fitness(ch) for ch in population]

    best_fitness_history = []
    avg_fitness_history = []

    # Gen 0 stats
    best_fitness_history.append(max(fitnesses))
    avg_fitness_history.append(sum(fitnesses) / len(fitnesses))

    # ---- Evolution loop ----
    for _ in range(1, generations + 1):

        # Elitism
        elite_indices = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True,
        )[:elitism]

        new_population: List[Tuple[int, ...]] = [population[i] for i in elite_indices]

        # Reproduction
        while len(new_population) < pop_size:
            parent1 = ga.tournament_select(population, fitnesses, tournament_k)
            parent2 = ga.tournament_select(population, fitnesses, tournament_k)

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = ga.uniform_crossover(parent1, parent2, forced_indices)
            else:
                child1, child2 = parent1, parent2

            # Mutation
            child1 = ga.mutate(child1, mutation_rate_gene, same_class_prob, forced_indices)
            if len(new_population) < pop_size:
                new_population.append(child1)

            if len(new_population) < pop_size:
                child2 = ga.mutate(child2, mutation_rate_gene, same_class_prob, forced_indices)
                new_population.append(child2)

        population = new_population
        fitnesses = [ga.chromosome_fitness(ch) for ch in population]

        best_fitness_history.append(max(fitnesses))
        avg_fitness_history.append(sum(fitnesses) / len(fitnesses))

    return best_fitness_history, avg_fitness_history


# ------------------------------------------------------------
# Main benchmark driver
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GA benchmark experiments for Clash Royale deck builder")

    parser.add_argument("--generations", type=int, default=50, help="Number of generations per experiment")
    parser.add_argument("--same-class-prob", type=float, default=0.8, help="Same class mutation probability")
    parser.add_argument("--crossover-rate", type=float, default=0.9, help="Crossover rate")
    parser.add_argument("--elitism", type=int, default=2, help="Number of elites to keep each generation")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility")
    parser.add_argument(
        "--force",
        nargs="*",
        default=[],
        help="List of card slugs that must appear in every deck (same for all experiments)",
    )

    # >>> widened ranges with same or smaller steps <<<
    parser.add_argument(
        "--mutation-grid",
        nargs="*",
        type=float,
        # from 0.01 up to 0.5, smaller steps on the low end
        default=[0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
        help="Mutation rates to benchmark",
    )
    parser.add_argument(
        "--popsize-grid",
        nargs="*",
        type=int,
        # from 50 up to 400, finer on smaller sizes
        default=[50, 75, 100, 150, 200, 300, 400],
        help="Population sizes to benchmark",
    )
    parser.add_argument(
        "--k-grid",
        nargs="*",
        type=int,
        # from weak to strong tournament pressure
        default=[2, 3, 4, 5, 7, 9],
        help="Tournament k values to benchmark",
    )

    args = parser.parse_args()

    base_seed = args.seed

    # ---------------- Experiment 1: mutation rate ----------------
    plt.figure()
    for i, mut in enumerate(args.mutation_grid):
        best_hist, _ = run_experiment(
            pop_size=100,
            generations=args.generations,
            tournament_k=3,
            crossover_rate=args.crossover_rate,
            mutation_rate_gene=mut,
            same_class_prob=args.same_class_prob,
            elitism=args.elitism,
            forced_slugs=args.force,
            seed=base_seed + i,
        )
        plt.plot(best_hist, label=f"mutation={mut}")

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Effect of mutation rate on best fitness")
    plt.legend()
    plt.tight_layout()

    # ---------------- Experiment 2: population size ----------------
    plt.figure()
    for i, pop in enumerate(args.popsize_grid):
        best_hist, _ = run_experiment(
            pop_size=pop,
            generations=args.generations,
            tournament_k=3,
            crossover_rate=args.crossover_rate,
            mutation_rate_gene=0.1,
            same_class_prob=args.same_class_prob,
            elitism=args.elitism,
            forced_slugs=args.force,
            seed=base_seed + 100 + i,
        )
        plt.plot(best_hist, label=f"pop={pop}")

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Effect of population size on best fitness")
    plt.legend()
    plt.tight_layout()

    # ---------------- Experiment 3: tournament k ----------------
    plt.figure()
    for i, k in enumerate(args.k_grid):
        best_hist, _ = run_experiment(
            pop_size=100,
            generations=args.generations,
            tournament_k=k,
            crossover_rate=args.crossover_rate,
            mutation_rate_gene=0.1,
            same_class_prob=args.same_class_prob,
            elitism=args.elitism,
            forced_slugs=args.force,
            seed=base_seed + 200 + i,
        )
        plt.plot(best_hist, label=f"k={k}")

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Effect of tournament size k on best fitness")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
