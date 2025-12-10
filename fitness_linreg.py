import json
import random
from pathlib import Path

import numpy as np
import torch
from verified_decks import deck_scores


CARDS_JSON = Path("cards.json")
COUNTER_MATRIX_NPY = Path("counter_matrix.npy")
SYNERGY_MATRIX_NPY = Path("synergy_matrix.npy")


def load_cards_and_indices(path: Path = CARDS_JSON):
    """
    Load cards.json and return:
      - cards: {slug: feature_dict}
      - feature_names: list of per-card feature names (excluding 'index')
      - slug_to_index: {slug: matrix_index}
    """
    with path.open("r", encoding="utf-8") as f:
        cards = json.load(f)

    if not isinstance(cards, dict):
        raise ValueError("cards.json must be a mapping {slug: {...}}")

    # Collect feature names from the first card (all except 'index')
    first_card = next(iter(cards.values()))
    feature_names = [k for k in first_card.keys() if k != "index"]

    # Build slug -> index mapping for matrices
    slug_to_index = {}
    for slug, info in cards.items():
        idx = info.get("index")
        if idx is None:
            raise ValueError(f"Card '{slug}' has no 'index' field in cards.json")
        slug_to_index[slug] = int(idx)

    return cards, feature_names, slug_to_index


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
    There are a few decks with names like ',goblin-barrel'.
    """
    return raw_name.strip().lstrip(",")


def deck_to_features_all(
    deck,
    cards,
    feature_names,
    slug_to_index,
    counter_matrix,
    synergy_matrix,
):
    """
    Build a feature vector for a deck using:

    1) Aggregated per-card features from cards.json
       For each feature_name f, we sum f over all cards in the deck.

    2) Additional engineered features:
       - num_synergy_pairs : number of unordered card pairs (i, j), i < j
                             with positive synergy (matrix value > 0)
       - total_counters    : sum over deck cards of how many cards they
                             counter (counter_matrix[i, j] > 0).
                             If multiple deck cards counter the same target,
                             it's counted multiple times.
    """
    slugs = [clean_slug(name) for name in deck]

    # ------------- base per-card features -------------
    base_feats = [0.0 for _ in feature_names]
    for slug in slugs:
        if slug not in cards:
            raise KeyError(f"Card '{slug}' not found in cards.json")
        card_info = cards[slug]
        for k, feat_name in enumerate(feature_names):
            base_feats[k] += float(card_info.get(feat_name, 0.0))

    # ------------- engineered features (synergy / counters) -------------
    # Map slugs to indices for matrices
    idxs = []
    for slug in slugs:
        if slug not in slug_to_index:
            raise KeyError(f"Card '{slug}' has no index in cards.json")
        idxs.append(slug_to_index[slug])

    # 1) Number of synergy pairs
    num_synergy_pairs = 0
    n = len(idxs)
    for a in range(n):
        i = idxs[a]
        for b in range(a + 1, n):
            j = idxs[b]
            if synergy_matrix[i, j] > 0:
                num_synergy_pairs += synergy_matrix[i, j]

    # 2) Total counters (out-degree for each deck card)
    total_counters = 0
    for i in idxs:
        total_counters += int(np.count_nonzero(counter_matrix[i, :] > 0))

    # Concatenate everything into one feature vector
    all_feats = base_feats + [
        float(num_synergy_pairs),
        float(total_counters),
    ]

    return all_feats


def build_dataset(cards, feature_names, slug_to_index, counter_matrix, synergy_matrix):
    """
    Build the (X, y) dataset for linear regression.

    X: deck features (base card features + engineered synergy/counter features)
    y: 100 * deck_score
    """
    X_list = []
    y_list = []

    for deck, score in deck_scores.items():
        feats = deck_to_features_all(
            deck,
            cards,
            feature_names,
            slug_to_index,
            counter_matrix,
            synergy_matrix,
        )
        X_list.append(feats)
        y_list.append(100.0 * float(score))

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    return X, y


def train_test_split(X, y, train_size=40, seed=42):
    """
    Simple shuffled train/test split by index.
    """
    n_samples = X.size(0)
    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def train_model(
    X_train,
    y_train,
    n_epochs=1200,
    lr=1e-2,
    weight_decay=1e-2,
    cost_index: int = None,
):
    """
    Train a simple linear regression model with constrained weights:

    - COST feature's weight is forced to be non-positive (<= 0).
    - Every other feature's weight is forced to be non-negative (>= 0).
    """
    if cost_index is None:
        raise ValueError("cost_index must be provided and point to the 'cost' feature.")

    in_features = X_train.size(1)
    model = torch.nn.Linear(in_features, 1, bias=True)
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # Constrain weights after each update
        with torch.no_grad():
            W = model.weight  # shape: (1, in_features)

            # mask for non-cost features
            mask = torch.ones_like(W, dtype=torch.bool)
            mask[0, cost_index] = False

            # IMPORTANT: advanced indexing returns a copy, so we must assign back.
            # All non-cost weights >= 0
            W[mask] = W[mask].clamp(min=0.0)

            # Cost weight <= 0 (but can be negative)
            W[0, cost_index].clamp_(max=0.0)

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Compute train and test MSE for a trained model.
    """
    criterion = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(X_train), y_train).item()
        test_loss = criterion(model(X_test), y_test).item()
    return train_loss, test_loss


def main():
    torch.manual_seed(42)

    cards, feature_names, slug_to_index = load_cards_and_indices()
    counter_matrix, synergy_matrix = load_matrices()
    X, y = build_dataset(cards, feature_names, slug_to_index, counter_matrix, synergy_matrix)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    # Extend feature_names list with the two extra features
    extra_feature_names = ["num_synergy_pairs", "total_counters"]
    all_feature_names = feature_names + extra_feature_names

    # Find index of the 'cost' feature so we can constrain its weight
    if "cost" not in all_feature_names:
        raise RuntimeError("'cost' feature not found in feature names; cannot enforce cost constraint.")
    cost_index = all_feature_names.index("cost")

    model = train_model(X_train, y_train, cost_index=cost_index)
    train_loss, test_loss = evaluate_model(model, X_train, y_train, X_test, y_test)

    print("Feature weights:")
    weights = model.weight.detach().numpy()[0]
    for name, val in zip(all_feature_names, weights):
        print(f"{name} : {val:+.6f}")

    print("\nBias:")
    print(f"bias : {model.bias.item():+.6f}")

    print(f"\nTrain MSE: {train_loss:.6f}")
    print(f"Test MSE:  {test_loss:.6f}")


if __name__ == "__main__":
    main()
