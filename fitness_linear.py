import json
from pathlib import Path
import numpy as np
import sys

CARDS_JSON = Path("cards.json")
COUNTER_MATRIX_NPY = Path("counter_matrix.npy")
SYNERGY_MATRIX_NPY = Path("synergy_matrix.npy")

# Fixed core weights
WEIGHTS = {
    "cost": -1,
    "tank": 1.5,
    "air": 1.0,
    "close-combat": 2.0,
    "far-combat": 1.0,
    "win-condition": 4.0,
    "big-atk-spell": 2.0,
    "small-atk-spell": 1.0,
    "def-spell": 1.0,
    "anti-air": 3.0,
    "building": 1.0,
    "spawn": 1.0,
    "swarm": 1.0,
    "anti-swarm": 2.0,
    "anti-tank": 2.0,
}


TARGET_MIN_COST = 20
TARGET_MAX_COST = 36
FEATURE_MIN_VALUE = 1.5


def load_cards_and_indices(path: Path = CARDS_JSON):
    with path.open("r", encoding="utf-8") as f:
        cards = json.load(f)

    slug_to_index = {}
    for slug, info in cards.items():
        slug_to_index[slug] = int(info["index"])

    return cards, slug_to_index


def load_matrices():
    counter_matrix = np.load(COUNTER_MATRIX_NPY)
    synergy_matrix = np.load(SYNERGY_MATRIX_NPY)
    return counter_matrix, synergy_matrix


def clean_slug(raw_name: str) -> str:
    return raw_name.strip().lstrip(",")


def deck_features_by_name(deck, cards, slug_to_index, counter_matrix, synergy_matrix):

    slugs = [clean_slug(name) for name in deck]

    base_feature_names = list(PENALTY_WEIGHTS.keys())

    features = {}
    for name in base_feature_names:
        features[name] = 0.0

    features["num_synergy_pairs"] = 0.0
    features["total_counters"] = 0.0

    for slug in slugs:
        card_info = cards[slug]
        for feat_name in base_feature_names:
            features[feat_name] += float(card_info.get(feat_name, 0.0))

    idxs = [slug_to_index[s] for s in slugs]

    num_synergy_pairs = 0.0
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            if synergy_matrix[idxs[a], idxs[b]] > 0:
                num_synergy_pairs += float(synergy_matrix[idxs[a], idxs[b]])

    total_counters = 0.0
    for i in idxs:
        total_counters += float(np.count_nonzero(counter_matrix[i, :] > 0))

    features["num_synergy_pairs"] = num_synergy_pairs
    features["total_counters"] = total_counters

    return features


# ------------------------------------------------------------------
# FITNESS FUNCTION
#
# - Only synergy and counters contribute positively
# - All other features act ONLY as penalties
# - Missing features are punished with their corresponding weight
# - Elixir cost is punished by distance from [20, 36]
# ------------------------------------------------------------------
def deck_fitness(feature_dict):

    # Base objective: maximize synergy and counters only
    fitness = (
        feature_dict["num_synergy_pairs"] * OBJECTIVE_WEIGHTS["num_synergy_pairs"]
        + feature_dict["total_counters"] * OBJECTIVE_WEIGHTS["total_counters"]
    )

    # Penalize missing utility features
    for name, weight in PENALTY_WEIGHTS.items():
        if name == "cost":
            continue

        value = float(feature_dict.get(name, 0.0))

        # If feature is completely absent in the deck, apply full penalty
        if value < FEATURE_MIN_VALUE:
            fitness -= weight

    # Penalize elixir cost outside allowed interval
    total_cost = feature_dict["cost"]
    cost_weight = PENALTY_WEIGHTS["cost"]

    if total_cost < TARGET_MIN_COST:
        fitness -= cost_weight * (TARGET_MIN_COST - total_cost)

    elif total_cost > TARGET_MAX_COST:
        fitness -= cost_weight * (total_cost - TARGET_MAX_COST)

    return fitness


# ------------------------------------------------------------------
# CLI HELPER FOR MANUAL DECK INSPECTION
#
# Usage:
#   python fitness.py card_slug_1 card_slug_2 ... card_slug_8
# ------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python fitness.py card_slug1 card_slug2 ...")
        sys.exit(1)

    deck_slugs = sys.argv[1:]

    cards, slug_to_index = load_cards_and_indices()
    counter_matrix, synergy_matrix = load_matrices()

    features = deck_features_by_name(
        deck_slugs,
        cards,
        slug_to_index,
        counter_matrix,
        synergy_matrix,
    )

    fitness_value = deck_fitness(features)

    print("Deck:")
    for slug in deck_slugs:
        print(f"  {slug}")

    print("\nComputed features:")
    for k in sorted(features.keys()):
        print(f"  {k:20s}: {features[k]}")

    print("\nFinal fitness value:")
    print(f"  {fitness_value}")
