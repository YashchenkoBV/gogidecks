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


ELITISM_K = 2


def slugs_to_indices(slugs: Sequence[str]) -> List[int]:
    return [_slug_to_index[s] for s in slugs]


def indices_to_slugs(indices: Sequence[int]) -> List[str]:
    return [_index_to_slug[i] for i in indices]


def normalize_chromosome(genes: Sequence[int]) -> Tuple[int, ...]:
    unique_sorted = tuple(sorted(set(genes)))
    if len(unique_sorted) != DECK_SIZE:
        raise ValueError("Invalid chromosome size")
    return unique_sorted


def chromosome_is_valid(chromosome: Sequence[int]) -> bool:
    slugs = indices_to_slugs(chromosome)
    class_counts = {}

    for slug in slugs:
        cls = _classes[slug]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    for cls, limit in CLASS_LIMITS.items():
        if class_counts.get(cls, 0) > limit:
            return False

    for cls in CLASS_LIMITS.keys():
        if class_counts.get(cls, 0) == 0:
            return False

    return True


def random_gene(exclude: Set[int] | None = None) -> int:
    if exclude is None:
        return random.randrange(NUM_CARDS)

    candidates = [i for i in range(NUM_CARDS) if i not in exclude]
    return random.choice(candidates)


def build_random_chromosome(forced_indices: Sequence[int] | None = None) -> Tuple[int, ...]:
    genes: Set[int] = set(forced_indices) if forced_indices else set()

    class_counts = {}
    for idx in genes:
        cls = _classes[_index_to_slug[idx]]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    while len(genes) < DECK_SIZE:
        candidate = random_gene(exclude=genes)
        cls = _classes[_index_to_slug[candidate]]

        if class_counts.get(cls, 0) < CLASS_LIMITS.get(cls, DECK_SIZE):
            genes.add(candidate)
            class_counts[cls] = class_counts.get(cls, 0) + 1

    chrom = normalize_chromosome(genes)

    if not chromosome_is_valid(chrom):
        return build_random_chromosome(forced_indices)

    return chrom


def chromosome_fitness(chromosome: Sequence[int]) -> float:
    slugs = indices_to_slugs(chromosome)
    features = deck_features_by_name(
        slugs,
        _cards,
        _slug_to_index,
        _counter_matrix,
        _synergy_matrix,
    )
    return deck_fitness(features)


def initial_population(pop_size: int, forced_slugs: Sequence[str] = ()):
    forced_indices = slugs_to_indices(forced_slugs)
    population = []

    while len(population) < pop_size:
        c = build_random_chromosome(forced_indices)
        population.append(c)

    return population


def tournament_select(population, fitnesses, k):
    competitors = random.sample(range(len(population)), k)
    best_idx = max(competitors, key=lambda i: fitnesses[i])
    return population[best_idx]


def uniform_crossover(parent1, parent2, forced_indices=()):
    child1 = build_random_chromosome(forced_indices)
    child2 = build_random_chromosome(forced_indices)
    return child1, child2


def mutate(chromosome, mrate, same_class_prob, forced_indices=()):
    if random.random() < mrate:
        return build_random_chromosome(forced_indices)
    return chromosome


# ✅ ✅ ✅ ELITIST GA LOOP
def run_ga(
    pop_size,
    generations,
    tournament_k,
    crossover_rate,
    mutation_rate_gene,
    same_class_prob,
    forced_slugs=(),
    seed=None,
):

    if seed is not None:
        random.seed(seed)

    forced_indices = slugs_to_indices(forced_slugs)

    population = initial_population(pop_size, forced_slugs)
    fitnesses = [chromosome_fitness(c) for c in population]

    for gen in range(generations):

        # ✅ ✅ ✅ ELITISM
        elite_indices = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True
        )[:ELITISM_K]

        new_population = [population[i] for i in elite_indices]

        while len(new_population) < pop_size:

            p1 = tournament_select(population, fitnesses, tournament_k)
            p2 = tournament_select(population, fitnesses, tournament_k)

            if random.random() < crossover_rate:
                c1, c2 = uniform_crossover(p1, p2, forced_indices)
            else:
                c1, c2 = p1, p2

            c1 = mutate(c1, mutation_rate_gene, same_class_prob, forced_indices)

            if len(new_population) < pop_size:
                new_population.append(c1)

            if len(new_population) < pop_size:
                c2 = mutate(c2, mutation_rate_gene, same_class_prob, forced_indices)
                new_population.append(c2)

        population = new_population
        fitnesses = [chromosome_fitness(c) for c in population]

        print(f"Gen {gen+1} | Best: {max(fitnesses):.4f}")

    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    return population, fitnesses, population[best_idx], fitnesses[best_idx]
