#!/usr/bin/env python3
import sys
import json
from pathlib import Path

import numpy as np


CARDS_JSON = Path("cards.json")


def load_slug_to_index() -> dict:
    if not CARDS_JSON.exists():
        print(f"cards.json not found at {CARDS_JSON.resolve()}")
        sys.exit(1)

    with CARDS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("cards.json must be an object {slug: {...}}")
        sys.exit(1)

    slug_to_index = {}
    for slug, info in data.items():
        idx = info.get("index")
        if idx is None:
            print(f"Warning: slug '{slug}' has no 'index' field, skipping.")
            continue
        slug_to_index[slug] = int(idx)

    if not slug_to_index:
        print("No valid slug -> index mappings found in cards.json")
        sys.exit(1)

    return slug_to_index


def main():
    if len(sys.argv) != 4:
        print("Usage:")
        print("  python3 check_npy_by_slug.py <file.npy> <slug_i> <slug_j>")
        print("Example:")
        print("  python3 check_npy_by_slug.py counter_matrix.npy little-prince goblin-barrel")
        sys.exit(1)

    npy_file = Path(sys.argv[1])
    slug_i = sys.argv[2]
    slug_j = sys.argv[3]

    if not npy_file.exists():
        print(f"File not found: {npy_file}")
        sys.exit(1)

    slug_to_index = load_slug_to_index()

    if slug_i not in slug_to_index:
        print(f"Slug '{slug_i}' not found in cards.json")
        sys.exit(1)
    if slug_j not in slug_to_index:
        print(f"Slug '{slug_j}' not found in cards.json")
        sys.exit(1)

    i = slug_to_index[slug_i]
    j = slug_to_index[slug_j]

    arr = np.load(npy_file)

    if i < 0 or i >= arr.shape[0] or j < 0 or j >= arr.shape[1]:
        print("Index out of bounds for matrix shape", arr.shape)
        print(f"{slug_i} -> {i}, {slug_j} -> {j}")
        sys.exit(1)

    value = arr[i, j]
    print(f"{npy_file.name}[{slug_i} ({i}), {slug_j} ({j})] = {value}")


if __name__ == "__main__":
    main()
