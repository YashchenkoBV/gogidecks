#!/usr/bin/env python3
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup, Tag, NavigableString

BASE_DETAIL_URL = "https://www.deckshop.pro/card/detail/"

CARDS_JSON = Path("cards.json")
COUNTER_MATRIX_NPY = Path("counter_matrix.npy")
SYNERGY_MATRIX_NPY = Path("synergy_matrix.npy")

REQUEST_DELAY_SEC = 0.5  # be polite


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0 Safari/537.36"
            ),
            "Accept-Language": "en,en-US;q=0.9",
        }
    )
    return s


def load_cards() -> Dict[str, dict]:
    if not CARDS_JSON.exists():
        raise SystemExit(f"{CARDS_JSON} not found")

    with CARDS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise SystemExit("cards.json must be an object {slug: {...}}")

    return data


def save_cards(cards: Dict[str, dict]) -> None:
    with CARDS_JSON.open("w", encoding="utf-8") as f:
        json.dump(cards, f, indent=2, ensure_ascii=False)
    print(f"Updated {CARDS_JSON.resolve()}")


def is_heading(tag: Tag) -> bool:
    if not isinstance(tag, Tag):
        return False
    name = (tag.name or "").lower()
    return name in {"h2", "h3", "h4", "h5", "h6"}


def classify_section(tag: Tag, card_name: str) -> str:
    """
    Decide if a heading is a counter or synergy section for this card.
    We only care about:
      - '<Name> can counter these ... cards'
      - '<Name> ... synergies'
    """
    text = tag.get_text(" ", strip=True).lower()
    base = card_name.lower()

    # Examples:
    # 'valkyrie can counter these cards 80/121'
    # 'valkyrie can counter these attacking cards 70/121'
    if base in text and "can counter these" in text:
        return "counter"

    # Examples:
    # 'valkyrie synergies 119/121'
    # 'valkyrie attack synergies 60/121'
    # 'valkyrie defend synergies 50/121'
    if base in text and "synergies" in text:
        return "synergy"

    return ""


def iter_section_card_links(start_heading: Tag) -> List[Tag]:
    """
    Starting from a heading, walk following siblings until the next heading.
    Collect all <a> tags that have data-id and href starting with /card/detail/.
    """
    links: List[Tag] = []

    for sib in start_heading.next_siblings:
        if isinstance(sib, NavigableString):
            continue
        if isinstance(sib, Tag):
            if is_heading(sib):
                break  # end of this section

            for a in sib.find_all("a", attrs={"data-id": True}, href=True):
                href = a["href"]
                if not href.startswith("/card/detail/"):
                    continue
                links.append(a)

    return links


def card_weight_from_link(a_tag: Tag) -> float:
    """
    Given:
      <a ...>
        <div class="opacity-25"> ... </div>
      </a>
    If class contains 'opacity-25' -> 0.25
    else -> 1.0
    """
    div = a_tag.find("div")
    if not div:
        return 1.0
    classes = div.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    if "opacity-25" in classes:
        return 0.25
    return 1.0


def slug_from_href(href: str) -> str:
    # e.g. /card/detail/barbarian-hut or /card/detail/barbarian-hut?foo=bar
    path = href.split("?", 1)[0]
    return path.rstrip("/").split("/")[-1]


def fetch_card_detail(session: requests.Session, slug: str) -> BeautifulSoup:
    url = BASE_DETAIL_URL + slug
    print(f"  - Fetching {url}")
    resp = session.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def derive_card_name_from_slug(slug: str) -> str:
    # 'little-prince' -> 'Little Prince'
    parts = slug.split("-")
    return " ".join(p.capitalize() for p in parts)


def build_matrices(session: requests.Session, cards: Dict[str, dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    cards: {slug: {...}}
    returns:
      counter_matrix, synergy_matrix
    and also updates cards[slug]["data_id"] when encountered.
    """
    slugs = sorted(cards.keys())
    n = len(slugs)

    # Assign indices
    slug_to_index: Dict[str, int] = {}
    for idx, slug in enumerate(slugs):
        slug_to_index[slug] = idx
        entry = cards.setdefault(slug, {})
        entry["index"] = idx

    counter_matrix = np.zeros((n, n), dtype=np.float32)
    synergy_matrix = np.zeros((n, n), dtype=np.float32)

    # Process each card detail page
    for slug in slugs:
        i = slug_to_index[slug]
        entry = cards[slug]
        card_name = entry.get("name") or derive_card_name_from_slug(slug)
        entry["name"] = card_name  # ensure name is present

        soup = fetch_card_detail(session, slug)
        time.sleep(REQUEST_DELAY_SEC)

        for h in soup.find_all(is_heading):
            sec_type = classify_section(h, card_name)
            if not sec_type:
                continue

            links = iter_section_card_links(h)
            if not links:
                continue

            for a in links:
                href = a.get("href") or ""
                target_slug = slug_from_href(href)
                if target_slug not in slug_to_index:
                    # This target card is not in cards.json (yet) – ignore
                    continue

                j = slug_to_index[target_slug]

                # data-id on target
                data_id = a.get("data-id")
                if data_id is not None:
                    try:
                        did = int(data_id)
                    except ValueError:
                        did = None
                    if did is not None:
                        t_entry = cards.setdefault(target_slug, {})
                        # don't overwrite if already set
                        if "data_id" not in t_entry:
                            t_entry["data_id"] = did

                w = card_weight_from_link(a)

                if sec_type == "counter":
                    # slug (i) can counter target_slug (j)
                    if w > counter_matrix[i, j]:
                        counter_matrix[i, j] = w
                elif sec_type == "synergy":
                    # slug (i) synergizes with target_slug (j)
                    if w > synergy_matrix[i, j]:
                        synergy_matrix[i, j] = w

    return counter_matrix, synergy_matrix


def main():
    cards = load_cards()
    session = get_session()

    print(f"Loaded {len(cards)} cards from {CARDS_JSON}")

    counter_matrix, synergy_matrix = build_matrices(session, cards)

    # Save matrices
    np.save(COUNTER_MATRIX_NPY, counter_matrix)
    np.save(SYNERGY_MATRIX_NPY, synergy_matrix)

    print(
        f"Saved counter matrix to {COUNTER_MATRIX_NPY.resolve()} "
        f"with shape {counter_matrix.shape}"
    )
    print(
        f"Saved synergy matrix to {SYNERGY_MATRIX_NPY.resolve()} "
        f"with shape {synergy_matrix.shape}"
    )

    # Save updated cards.json (with indices and discovered data_id)
    save_cards(cards)


if __name__ == "__main__":
    main()
