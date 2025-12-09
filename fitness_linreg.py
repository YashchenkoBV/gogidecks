import json
import random
import torch
from verified_decks import deck_scores


def load_cards(path="cards.json"):
    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    first_card = next(iter(cards.values()))
    feature_names = [k for k in first_card.keys() if k != "index"]

    return cards, feature_names


def deck_to_features(deck, cards, feature_names):
    features = [0.0] * len(feature_names)

    for raw_name in deck:
        name = raw_name.strip().lstrip(",")

        if name not in cards:
            raise KeyError(f"Card '{name}' not found in cards.json")

        card_data = cards[name]
        for i, feat_name in enumerate(feature_names):
            features[i] += float(card_data[feat_name])

    return features


def build_dataset(cards, feature_names):
    X_list = []
    y_list = []

    for deck, score in deck_scores.items():
        X_list.append(deck_to_features(deck, cards, feature_names))
        y_list.append(100 * float(score))

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

    return X, y


def train_test_split(X, y, train_size=40, seed=42):
    n_samples = X.size(0)
    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main():
    torch.manual_seed(42)

    cards, feature_names = load_cards()
    X, y = build_dataset(cards, feature_names)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    model = torch.nn.Linear(X_train.size(1), 1, bias=True)
    criterion = torch.nn.MSELoss()

    # ✅ L2 REGULARIZATION
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
        weight_decay=1e-2
    )

    n_epochs = 1200
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # ✅ CONSTRAINT PROJECTION
        with torch.no_grad():
            # Enforce all weights >= 0
            model.weight[:, 1:].clamp_(min=0.0)

            # Enforce COST weight <= 0 (index 0)
            model.weight[:, 0].clamp_(max=0.0)

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(X_train), y_train).item()
        test_loss = criterion(model(X_test), y_test).item()

    print("Feature names (order):")
    print(feature_names)

    print("\nFinal constrained weights:")
    print(model.weight.detach().numpy())

    print("\nBias:")
    print(model.bias.detach().numpy())

    print(f"\nTrain MSE: {train_loss:.6f}")
    print(f"Test MSE:  {test_loss:.6f}")


if __name__ == "__main__":
    main()
