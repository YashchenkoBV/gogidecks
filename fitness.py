import json
import sys
from pathlib import Path

import numpy as np

# Paths (adjust if needed)
CARDS_JSON = Path("cards.json")
COUNTER_MATRIX_NPY = Path("counter_matrix.npy")
SYNERGY_MATRIX_NPY = Path("synergy_matrix.npy")

# Fixed core weights
WEIGHTS = {
    "cost": -0.1875,
    "tank": 1.5,
    "air": 1.0,
    "close-combat": 2.0,
    "far-combat": 1.0,
    "win-condition": 3.0,
    "big-atk-spell": 0.5,
    "small-atk-spell": 0.5,
    "def-spell": 1.0,
    "anti-air": 2.0,
    "building": 1.0,
    "spawn": 0.5,
    "swarm": 0.5,
    "anti-swarm": 1.5,
    "anti-tank": 2.0,
    "num_synergy_pairs": 0.5,
    "total_counters": 0.01,
}


def load_cards_and_indices(path: Path = CARDS_JSON):
    """
    Load cards.json and return:
      - cards: {slug: feature_dict}
      - slug_to_index: {slug: matrix_index}
    """
    with path.open("r", encoding="utf-8") as f:
        cards = json.load(f)

    if not isinstance(cards, dict):
        raise ValueError("cards.json must be a mapping {slug: {...}}")

    slug_to_index = {}
    for slug, info in cards.items():
        idx = info.get("index")
        if idx is None:
            raise ValueError(f"Card '{slug}' has no 'index' field in cards.json")
        slug_to_index[slug] = int(idx)

    return cards, slug_to_index


def load_matrices():
    """
    Load counter and synergy matrices from .npy files.
    """
    if not COUNTER_MATRIX_NPY.exists():
        raise FileNotFoundError(f"{COUNTER_MATRIX_NPY} not found")
    if not SYNERGY_MATRIX_NPY.exists():
        raise FileNotFoundError(f"{SYNERGY_MATRIX_NPY} not found")

    counter_matrix = np.load(COUNTER_MATRIX_NPY)
    synergy_matrix = np.load(SYNERGY_MATRIX_NPY)

    if counter_matrix.shape != synergy_matrix.shape:
        raise ValueError(
            f"Shape mismatch: counter_matrix {counter_matrix.shape}, "
            f"synergy_matrix {synergy_matrix.shape}"
        )

    return counter_matrix, synergy_matrix


def clean_slug(raw_name: str) -> str:
    """
    Normalize deck card names to card slugs used in cards.json.
    """
    return raw_name.strip().lstrip(",")


def deck_features_by_name(
    deck,
    cards,
    slug_to_index,
    counter_matrix,
    synergy_matrix,
):
    """
    Compute feature values for a deck as a dict {feature_name: value} using
    only the features that appear in WEIGHTS.
    """
    slugs = [clean_slug(name) for name in deck]

    # Base per-card features (everything in WEIGHTS except engineered ones)
    base_feature_names = [
        name for name in WEIGHTS.keys()
        if name not in ("num_synergy_pairs", "total_counters")
    ]

    features = {name: 0.0 for name in WEIGHTS.keys()}

    # Aggregate per-card features
    for slug in slugs:
        if slug not in cards:
            raise KeyError(f"Card '{slug}' not found in cards.json")
        card_info = cards[slug]
        for feat_name in base_feature_names:
            features[feat_name] += float(card_info.get(feat_name, 0.0))

    # Engineered features need matrix indices
    idxs = []
    for slug in slugs:
        if slug not in slug_to_index:
            raise KeyError(f"Card '{slug}' has no index in cards.json")
        idxs.append(slug_to_index[slug])

    # num_synergy_pairs
    num_synergy_pairs = 0.0
    n = len(idxs)
    for a in range(n):
        i = idxs[a]
        for b in range(a + 1, n):
            j = idxs[b]
            if synergy_matrix[i, j] > 0:
                num_synergy_pairs += float(synergy_matrix[i, j])

    # total_counters
    total_counters = 0.0
    for i in idxs:
        total_counters += float(np.count_nonzero(counter_matrix[i, :] > 0))

    features["num_synergy_pairs"] = num_synergy_pairs
    features["total_counters"] = total_counters

    return features


def deck_fitness(feature_dict):
    """
    Compute fitness as sum(feature_value * weight) over all WEIGHTS.
    """
    fitness = 0.0
    for name, weight in WEIGHTS.items():
        val = float(feature_dict.get(name, 0.0))
        fitness += val * weight
    return fitness


def main():
    if len(sys.argv) < 2:
        print("Usage: python fitness.py card_slug1 card_slug2 ...")
        sys.exit(1)

    deck = sys.argv[1:]  # list of slugs from command line

    cards, slug_to_index = load_cards_and_indices()
    counter_matrix, synergy_matrix = load_matrices()

    features = deck_features_by_name(
        deck,
        cards,
        slug_to_index,
        counter_matrix,
        synergy_matrix,
    )
    fitness = deck_fitness(features)

    print("Deck:", ", ".join(deck))
    print("\nFeature values:")
    for name in WEIGHTS.keys():
        print(f"{name} : {features[name]:.3f}")

    print(f"\nFitness : {fitness:.3f}")


if __name__ == "__main__":
    main()
