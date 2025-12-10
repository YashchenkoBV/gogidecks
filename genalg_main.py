# genalg_main.py

from typing import Sequence, Tuple, List, Set
import random
import json

from fitness import (
    load_cards_and_indices,
    load_matrices,
    deck_features_by_name,
    deck_fitness,
)

_cards, _slug_to_index = load_cards_and_indices()
with open("classes.json", "r", encoding="utf-8") as f:
    _classes = json.load(f)

_counter_matrix, _synergy_matrix = load_matrices()

_index_to_slug = {idx: slug for slug, idx in _slug_to_index.items()}

NUM_CARDS = len(_index_to_slug)
DECK_SIZE = 8

CLASS_LIMITS = {
    "spell": 3,
    "troop": 6,
    "anti-building": 3,
    "building": 2,
}


# ============================================================
# Basic helpers
# ============================================================

def slugs_to_indices(slugs: Sequence[str]) -> List[int]:
    """Convert card slugs to integer indices."""
    return [_slug_to_index[s] for s in slugs]


def indices_to_slugs(indices: Sequence[int]) -> List[str]:
    """Convert integer indices to card slugs."""
    return [_index_to_slug[i] for i in indices]


def normalize_chromosome(genes: Sequence[int]) -> Tuple[int, ...]:
    """
    Normalize a chromosome:
    - remove duplicates
    - sort indices
    - ensure deck size is exactly DECK_SIZE
    """
    unique_sorted = tuple(sorted(set(genes)))
    if len(unique_sorted) != DECK_SIZE:
        raise ValueError("Invalid chromosome size")
    return unique_sorted


def chromosome_is_valid(chromosome: Sequence[int]) -> bool:
    """
    Check hard constraints:
    - obey CLASS_LIMITS
    - all cards recognized
    """
    slugs = indices_to_slugs(chromosome)
    class_counts = {}

    for slug in slugs:
        card_class = _classes[slug]   # correct source
        class_counts[card_class] = class_counts.get(card_class, 0) + 1

    for cls, limit in CLASS_LIMITS.items():
        if class_counts.get(cls, 0) > limit:
            return False

    return True


def random_gene(exclude: Set[int] | None = None) -> int:
    """
    Sample a random card index, optionally excluding a given set.
    """
    if exclude is None:
        return random.randrange(NUM_CARDS)

    candidates = [i for i in range(NUM_CARDS) if i not in exclude]
    if not candidates:
        raise RuntimeError("No available genes left to sample from.")
    return random.choice(candidates)


def build_random_chromosome(
    forced_indices: Sequence[int] | None = None,
) -> Tuple[int, ...]:
    """
    Build a random valid chromosome that:
    - contains all forced_indices
    - obeys CLASS_LIMITS
    - has exactly DECK_SIZE unique cards
    """
    genes: Set[int] = set(forced_indices) if forced_indices else set()

    # Count the classes for forced genes
    class_counts = {}
    for idx in genes:
        slug = _index_to_slug[idx]
        cls = _classes[slug]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    while len(genes) < DECK_SIZE:
        candidate = random_gene(exclude=genes)
        slug = _index_to_slug[candidate]
        cls = _classes[slug]

        # CHECK LIMIT BEFORE ADDING
        if class_counts.get(cls, 0) >= CLASS_LIMITS.get(cls, DECK_SIZE):
            continue

        genes.add(candidate)
        class_counts[cls] = class_counts.get(cls, 0) + 1

    chrom = normalize_chromosome(genes)

    if not chromosome_is_valid(chrom):
        # If something went wrong (very unlikely), try again recursively
        return build_random_chromosome(forced_indices)

    return chrom


# ============================================================
# Fitness wrappers
# ============================================================

def chromosome_fitness(chromosome: Sequence[int]) -> float:
    """
    Compute scalar fitness for a chromosome (deck) using the deck_* logic.
    """
    slugs = indices_to_slugs(chromosome)

    features = deck_features_by_name(
        slugs,
        _cards,
        _slug_to_index,
        _counter_matrix,
        _synergy_matrix,
    )

    return deck_fitness(features)


def chromosome_fitness_with_features(chromosome: Sequence[int]):
    """
    Debug version: returns (fitness, feature_dict)
    """
    slugs = indices_to_slugs(chromosome)

    features = deck_features_by_name(
        slugs,
        _cards,
        _slug_to_index,
        _counter_matrix,
        _synergy_matrix,
    )
    fit = deck_fitness(features)
    return fit, features


# ============================================================
# Selection, crossover, mutation
# ============================================================

def tournament_select(
    population: List[Tuple[int, ...]],
    fitnesses: List[float],
    k: int,
) -> Tuple[int, ...]:
    """
    k-way tournament selection.
    Returns a parent chromosome.
    """
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def uniform_crossover(
    parent1: Tuple[int, ...],
    parent2: Tuple[int, ...],
    forced_indices: Sequence[int] = (),
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Uniform crossover with set-based repair:
    - build children from union of parent genes
    - random selection until DECK_SIZE genes
    - ensure validity
    - forced indices always preserved
    - hard constraint enforcement
    """

    forced_set = set(forced_indices)
    set1, set2 = set(parent1), set(parent2)
    union = list(set1 | set2)

    attempts = 0
    MAX_ATTEMPTS = 100

    while True:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            # fallback: totally regenerate if repair fails
            c1 = build_random_chromosome(forced_indices)
            c2 = build_random_chromosome(forced_indices)
            return c1, c2

        # Start each child with forced genes
        genes1 = set(forced_set)
        genes2 = set(forced_set)

        # Fill the rest randomly from union
        shuffled = union[:]
        random.shuffle(shuffled)

        for g in shuffled:
            if len(genes1) < DECK_SIZE:
                genes1.add(g)
            if len(genes2) < DECK_SIZE:
                genes2.add(g)
            if len(genes1) >= DECK_SIZE and len(genes2) >= DECK_SIZE:
                break

        child1 = normalize_chromosome(genes1)
        child2 = normalize_chromosome(genes2)

        if chromosome_is_valid(child1) and chromosome_is_valid(child2):
            return child1, child2


def mutate(
    chromosome: Tuple[int, ...],
    mutation_rate_gene: float,
    same_class_prob: float,
    forced_indices: Sequence[int] = (),
) -> Tuple[int, ...]:
    """
    Gene-wise mutation:
    - Each gene mutates with probability mutation_rate_gene
    - Replacement comes from same class with probability same_class_prob
    - Forced genes are NEVER mutated
    - Always normalized
    - Always re-validated; if broken, rebuild a fresh random chromosome
    """
    forced_set = set(forced_indices)
    genes = list(chromosome)

    changed = False

    for i, old_gene in enumerate(genes):
        if old_gene in forced_set:
            continue

        if random.random() >= mutation_rate_gene:
            continue

        old_slug = _index_to_slug[old_gene]
        old_class = _classes[old_slug]

        if random.random() < same_class_prob:
            candidates = [
                idx
                for idx, slug in _index_to_slug.items()
                if _classes[slug] == old_class and idx not in genes
            ]
            if candidates:
                new_gene = random.choice(candidates)
            else:
                new_gene = random_gene(exclude=set(genes))
        else:
            new_gene = random_gene(exclude=set(genes))

        genes[i] = new_gene
        changed = True

    if not changed:
        return chromosome

    try:
        mutated = normalize_chromosome(genes)
    except ValueError:
        return build_random_chromosome(forced_indices)

    if not chromosome_is_valid(mutated):
        return build_random_chromosome(forced_indices)

    return mutated


# ============================================================
# GA driver
# ============================================================

def initial_population(
    pop_size: int,
    forced_slugs: Sequence[str] = (),
) -> List[Tuple[int, ...]]:
    """
    Create the initial population of size pop_size, each deck:
    - obeys CLASS_LIMITS
    - contains all forced_slugs
    """
    forced_indices = slugs_to_indices(forced_slugs)
    population: List[Tuple[int, ...]] = []

    while len(population) < pop_size:
        chrom = build_random_chromosome(forced_indices)
        population.append(chrom)

    return population


def run_ga(
    pop_size: int,
    generations: int,
    tournament_k: int,
    crossover_rate: float,
    mutation_rate_gene: float,
    same_class_prob: float,
    elitism: int,
    forced_slugs: Sequence[str] = (),
    seed: int | None = None,
):
    """
    Run the genetic algorithm.
    Returns (final_population, final_fitnesses, best_chromosome, best_fitness).
    """
    if seed is not None:
        random.seed(seed)

    forced_indices = slugs_to_indices(forced_slugs)

    # ---- Initial population ----
    population = initial_population(pop_size, forced_slugs)
    pop_size = len(population)
    fitnesses = [chromosome_fitness(ch) for ch in population]

    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best_chrom = population[best_idx]
    best_fitness = fitnesses[best_idx]

    print(f"Gen 0 | Best fitness: {best_fitness:.4f}")

    # ---- Evolution loop ----
    for gen in range(1, generations + 1):

        # ---- Elitism ----
        elite_indices = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True
        )[:elitism]

        new_population: List[Tuple[int, ...]] = [population[i] for i in elite_indices]

        # ---- Reproduction ----
        while len(new_population) < pop_size:

            parent1 = tournament_select(population, fitnesses, tournament_k)
            parent2 = tournament_select(population, fitnesses, tournament_k)

            if random.random() < crossover_rate:
                child1, child2 = uniform_crossover(parent1, parent2, forced_indices)
            else:
                child1, child2 = parent1, parent2

            child1 = mutate(child1, mutation_rate_gene, same_class_prob, forced_indices)
            new_population.append(child1)

            if len(new_population) < pop_size:
                child2 = mutate(child2, mutation_rate_gene, same_class_prob, forced_indices)
                new_population.append(child2)

        population = new_population
        fitnesses = [chromosome_fitness(ch) for ch in population]

        current_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        current_best_fitness = fitnesses[current_best_idx]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chrom = population[current_best_idx]

        print(f"Gen {gen} | Best fitness: {best_fitness:.4f}")

    return population, fitnesses, best_chrom, best_fitness


# ============================================================
# CLI front-end
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--tournament-k", type=int, default=20)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--same-class-prob", type=float, default=0.8)
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--force",
        nargs="*",
        default=[],
        help="List of card slugs that must appear in every deck",
    )

    args = parser.parse_args()

    # ---- Run GA ----
    final_population, final_fitnesses, best_deck, best_fitness = run_ga(
        pop_size=args.pop_size,
        generations=args.generations,
        tournament_k=args.tournament_k,
        crossover_rate=args.crossover_rate,
        mutation_rate_gene=args.mutation_rate,
        same_class_prob=args.same_class_prob,
        elitism=args.elitism,
        forced_slugs=args.force,
        seed=args.seed,
    )

    # Select up to top 3 unique decks (by chromosome content)
    sorted_indices = sorted(
        range(len(final_population)),
        key=lambda i: final_fitnesses[i],
        reverse=True
    )

    unique_best_indices = []
    seen_chromosomes = set()

    for idx in sorted_indices:
        chrom = tuple(final_population[idx])
        if chrom in seen_chromosomes:
            continue
        seen_chromosomes.add(chrom)
        unique_best_indices.append(idx)
        if len(unique_best_indices) >= 3:
            break

    print("\n==============================")
    print("TOP UNIQUE DECKS FOUND")
    print("==============================")

    for rank, idx in enumerate(unique_best_indices, start=1):
        chrom = final_population[idx]
        slugs = indices_to_slugs(chrom)
        fit = final_fitnesses[idx]

        print(f"\n#{rank}")
        print("Indices :", chrom)
        print("Slugs   :", " ".join(slugs))
        print(f"Fitness : {fit:.4f}")


if __name__ == "__main__":
    main()
