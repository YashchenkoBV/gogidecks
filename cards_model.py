# cards_model.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import json


@dataclass(frozen=True)
class Card:
    """Immutable representation of a Clash Royale card."""
    index: int
    slug: str
    card_class: str  # e.g. "troop", "spell", "anti-building"

    cost: int
    tank: float
    air: float
    close_combat: float
    far_combat: float
    win_condition: float
    big_atk_spell: float
    small_atk_spell: float
    def_spell: float
    anti_air: float
    building: float
    spawn: float
    swarm: float
    anti_swarm: float
    anti_tank: float


def _data_dir() -> Path:
    # Assumes cards_model.py lives in the same directory as cards.json / classes.json
    return Path(__file__).resolve().parent


def _load_raw_json() -> tuple[Dict[str, dict], Dict[str, str]]:
    cards_path = _data_dir() / "cards.json"
    classes_path = _data_dir() / "classes.json"

    with cards_path.open("r", encoding="utf-8") as f:
        cards_raw: Dict[str, dict] = json.load(f)

    with classes_path.open("r", encoding="utf-8") as f:
        classes_raw: Dict[str, str] = json.load(f)

    # Basic consistency checks
    cards_slugs = set(cards_raw.keys())
    classes_slugs = set(classes_raw.keys())

    missing_classes = cards_slugs - classes_slugs
    extra_classes = classes_slugs - cards_slugs

    if missing_classes or extra_classes:
        raise ValueError(
            f"Inconsistent cards/classes JSON.\n"
            f"Missing classes for: {sorted(missing_classes)}\n"
            f"Extra classes for unknown cards: {sorted(extra_classes)}"
        )

    return cards_raw, classes_raw


def _build_cards() -> List[Card]:
    cards_raw, classes_raw = _load_raw_json()

    cards: List[Card] = []
    for slug, attrs in cards_raw.items():
        card_class = classes_raw[slug]

        card = Card(
            index=attrs["index"],
            slug=slug,
            card_class=card_class,
            cost=attrs["cost"],
            tank=attrs["tank"],
            air=attrs["air"],
            close_combat=attrs["close-combat"],
            far_combat=attrs["far-combat"],
            win_condition=attrs["win-condition"],
            big_atk_spell=attrs["big-atk-spell"],
            small_atk_spell=attrs["small-atk-spell"],
            def_spell=attrs["def-spell"],
            anti_air=attrs["anti-air"],
            building=attrs["building"],
            spawn=attrs["spawn"],
            swarm=attrs["swarm"],
            anti_swarm=attrs["anti-swarm"],
            anti_tank=attrs["anti-tank"],
        )
        cards.append(card)

    # Sort by index and validate indices
    cards.sort(key=lambda c: c.index)
    indices = [c.index for c in cards]
    expected = list(range(len(cards)))
    if indices != expected:
        raise ValueError(
            f"Card indices are not contiguous 0..N-1. "
            f"Got {indices[0]}..{indices[-1]} with gaps or duplicates."
        )

    return cards


# ---- Global data structures ----

ALL_CARDS: List[Card] = _build_cards()

# Index-based lookups
IDX2CARD: List[Card] = ALL_CARDS
IDX2SLUG: List[str] = [c.slug for c in ALL_CARDS]

# Slug-based lookups
SLUG2CARD: Dict[str, Card] = {c.slug: c for c in ALL_CARDS}
SLUG2IDX: Dict[str, int] = {c.slug: c.index for c in ALL_CARDS}

# Class-based indices
CLASS2INDICES: Dict[str, List[int]] = {}
for card in ALL_CARDS:
    CLASS2INDICES.setdefault(card.card_class, []).append(card.index)


# ---- Helper functions ----

def all_indices() -> range:
    """Return a range over all valid card indices."""
    return range(len(ALL_CARDS))


def get_card_by_index(index: int) -> Card:
    return IDX2CARD[index]


def get_card_by_slug(slug: str) -> Card:
    return SLUG2CARD[slug]


def indices_to_slugs(indices: Sequence[int]) -> List[str]:
    return [IDX2SLUG[i] for i in indices]


def slugs_to_indices(slugs: Sequence[str]) -> List[int]:
    return [SLUG2IDX[s] for s in slugs]


__all__ = [
    "Card",
    "ALL_CARDS",
    "IDX2CARD",
    "SLUG2CARD",
    "IDX2SLUG",
    "SLUG2IDX",
    "CLASS2INDICES",
    "all_indices",
    "get_card_by_index",
    "get_card_by_slug",
    "indices_to_slugs",
    "slugs_to_indices",
]
