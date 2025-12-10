# ga_main.py

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

# Build inverse index -> slug mapping
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
# UTILITIES
# ============================================================

def slugs_to_indices(slugs: Sequence[str]) -> List[int]:
    """Convert list of card slugs to list of indices."""
    return [_slug_to_index[s] for s in slugs]


def indices_to_slugs(indices: Sequence[int]) -> List[str]:
    """Convert list of indices to list of slugs."""
    return [_index_to_slug[i] for i in indices]


def normalize_chromosome(genes: Sequence[int]) -> Tuple[int, ...]:
    """
    Enforce:
    - unique genes
    - sorted
    - exactly DECK_SIZE cards
    """
    unique_sorted = tuple(sorted(set(genes)))

    if len(unique_sorted) != DECK_SIZE:
        raise ValueError(
            f"Invalid chromosome size after normalization: {len(unique_sorted)} != {DECK_SIZE}"
        )

    return unique_sorted


def chromosome_is_valid(chromosome: Sequence[int]) -> bool:
    slugs = indices_to_slugs(chromosome)

    class_counts = {}

    for slug in slugs:
        card_class = _classes[slug]   # ✅ correct source
        class_counts[card_class] = class_counts.get(card_class, 0) + 1

    for cls, limit in CLASS_LIMITS.items():
        if class_counts.get(cls, 0) > limit:
            return False

    return True


def random_gene(exclude: Set[int] | None = None) -> int:
    """Sample a random card index not in exclude."""
    if exclude is None:
        return random.randrange(NUM_CARDS)

    candidates = [i for i in range(NUM_CARDS) if i not in exclude]
    return random.choice(candidates)


def build_random_chromosome(forced_indices: Sequence[int] | None = None) -> Tuple[int, ...]:
    """
    Build a valid chromosome of length 8 while RESPECTING class limits during construction.
    """
    genes: Set[int] = set(forced_indices) if forced_indices else set()

    # Count classes from forced cards
    class_counts = {}
    for idx in genes:
        slug = _index_to_slug[idx]
        cls = _classes[slug]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    if len(genes) > DECK_SIZE:
        raise ValueError("Too many forced cards for one deck")

    attempts = 0
    MAX_ATTEMPTS = 200

    while len(genes) < DECK_SIZE:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            # Restart completely if stuck
            return build_random_chromosome(forced_indices)

        candidate = random_gene(exclude=genes)
        slug = _index_to_slug[candidate]
        cls = _classes[slug]

        # ✅ CHECK LIMIT BEFORE ADDING
        if class_counts.get(cls, 0) < CLASS_LIMITS.get(cls, DECK_SIZE):
            genes.add(candidate)
            class_counts[cls] = class_counts.get(cls, 0) + 1

    return normalize_chromosome(genes)


def chromosome_fitness(chromosome: Sequence[int]) -> float:
    """
    Compute fitness directly from chromosome indices.
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

    fitness = deck_fitness(features)
    return fitness, features


# ============================================================
# INITIAL POPULATION
# ============================================================

def initial_population(pop_size: int, forced_slugs: Sequence[str] = ()) -> List[Tuple[int, ...]]:
    forced_indices = slugs_to_indices(forced_slugs)
    population: Set[Tuple[int, ...]] = set()

    safeguard = 0
    MAX_ATTEMPTS = pop_size * 200  # increased safely

    while len(population) < pop_size and safeguard < MAX_ATTEMPTS:
        chrom = build_random_chromosome(forced_indices)
        population.add(chrom)
        safeguard += 1

    if len(population) < pop_size:
        print(
            f"⚠️ WARNING: Only {len(population)} unique valid decks could be generated "
            f"with the given forced cards. Reducing population size accordingly."
        )

    return list(population)



# ============================================================
# TOURNAMENT SELECTION
# ============================================================

def tournament_select(
    population: List[Tuple[int, ...]],
    fitnesses: List[float],
    k: int,
) -> Tuple[int, ...]:
    """
    Select one chromosome via k-way tournament.
    """
    if k > len(population):
        raise ValueError("Tournament size k cannot exceed population size")

    competitors = random.sample(range(len(population)), k)
    best_idx = max(competitors, key=lambda i: fitnesses[i])

    return population[best_idx]

# ============================================================
# CROSSOVER
# ============================================================

def uniform_crossover(
    parent1: Tuple[int, ...],
    parent2: Tuple[int, ...],
    forced_indices: Sequence[int] = (),
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Uniform set-based crossover with:
    - equal probability from both parents
    - second child gets the complement
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

        # --- Child 1: 50/50 sampling from union ---
        random.shuffle(union)
        child1 = set(forced_set)
        for gene in union:
            if len(child1) >= DECK_SIZE:
                break
            if random.random() < 0.5:
                child1.add(gene)

        # Fill if short
        while len(child1) < DECK_SIZE:
            child1.add(random_gene(exclude=child1))

        child1 = normalize_chromosome(child1)

        # --- Child 2: complementary genes ---
        child2 = set(forced_set)
        for gene in union:
            if gene not in child1 and len(child2) < DECK_SIZE:
                child2.add(gene)

        while len(child2) < DECK_SIZE:
            child2.add(random_gene(exclude=child2))

        child2 = normalize_chromosome(child2)

        # --- Hard constraint enforcement ---
        if chromosome_is_valid(child1) and chromosome_is_valid(child2):
            return child1, child2


# ============================================================
# MUTATION
# ============================================================

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
    - Always constraint-validated with regeneration fallback
    """

    forced_set = set(forced_indices)
    genes = list(chromosome)

    attempts = 0
    MAX_ATTEMPTS = 50

    while True:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            # Emergency fallback: full random rebuild
            return build_random_chromosome(forced_indices)

        mutated = genes.copy()
        used = set(mutated)

        for i, gene in enumerate(mutated):

            # --- Forced genes cannot mutate ---
            if gene in forced_set:
                continue

            if random.random() < mutation_rate_gene:
                slug = _index_to_slug[gene]
                original_class = _classes[slug]

                # --- Choose mutation class ---
                if (
                    random.random() < same_class_prob
                    and original_class in CLASS_LIMITS
                ):
                    target_class = original_class
                else:
                    target_class = random.choice(list(CLASS_LIMITS.keys()))

                # --- Candidate pool from desired class ---
                candidates = [
                    _slug_to_index[s]
                    for s, cls in _classes.items()
                    if cls == target_class and _slug_to_index[s] not in used
                ]

                # --- Fallback if class exhausted ---
                if not candidates:
                    candidates = [
                        i for i in range(NUM_CARDS) if i not in used
                    ]

                new_gene = random.choice(candidates)

                used.remove(gene)
                used.add(new_gene)
                mutated[i] = new_gene

        # --- Normalize ---
        try:
            mutated_norm = normalize_chromosome(mutated)
        except ValueError:
            continue

        # --- Enforce hard constraints ---
        if chromosome_is_valid(mutated_norm):
            return mutated_norm


# ============================================================
# GENETIC ALGORITHM LOOP
# ============================================================

def run_ga(
    pop_size: int,
    generations: int,
    tournament_k: int,
    crossover_rate: float,
    mutation_rate_gene: float,
    same_class_prob: float,
    elitism: int = 1,
    forced_slugs: Sequence[str] = (),
    seed: int | None = None,
):
    """
    Minimal, safe, readable GA loop.
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

        new_population = [population[i] for i in elite_indices]

        # ---- Reproduction ----
        while len(new_population) < pop_size:

            parent1 = tournament_select(population, fitnesses, tournament_k)
            parent2 = tournament_select(population, fitnesses, tournament_k)

            # ---- Crossover ----
            if random.random() < crossover_rate:
                child1, child2 = uniform_crossover(
                    parent1, parent2, forced_indices
                )
            else:
                child1, child2 = parent1, parent2

            # ---- Mutation ----
            child1 = mutate(
                child1,
                mutation_rate_gene,
                same_class_prob,
                forced_indices,
            )

            if len(new_population) < pop_size:
                new_population.append(child1)

            if len(new_population) < pop_size:
                child2 = mutate(
                    child2,
                    mutation_rate_gene,
                    same_class_prob,
                    forced_indices,
                )
                new_population.append(child2)

        # ---- Replace population ----
        population = new_population
        fitnesses = [chromosome_fitness(ch) for ch in population]

        # ---- Track global best ----
        gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_chrom = population[gen_best_idx]

        # ---- Logging ----
        print(f"Gen {gen} | Best fitness: {best_fitness:.4f}")

    return population, fitnesses, best_chrom, best_fitness


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Genetic Algorithm Clash Royale Deck Builder")

    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
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

    top3_idx = sorted(
        range(len(final_population)),
        key=lambda i: final_fitnesses[i],
        reverse=True
    )[:3]

    print("\n==============================")
    print("🏆 TOP 3 DECKS FOUND")
    print("==============================")

    for rank, idx in enumerate(top3_idx, start=1):
        chrom = final_population[idx]
        slugs = indices_to_slugs(chrom)
        fit = final_fitnesses[idx]

        print(f"\n#{rank}")
        print("Indices :", chrom)
        print("Slugs   :", ", ".join(slugs))
        print(f"Fitness : {fit:.4f}")


if __name__ == "__main__":
    main()
