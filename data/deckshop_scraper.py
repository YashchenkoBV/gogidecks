#!/usr/bin/env python3
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

    # Normalize: ensure each slug maps to a dict
    normalized: Dict[str, dict] = {}
    for slug, info in data.items():
        if info is None:
            info = {}
        if not isinstance(info, dict):
            info = {}
        normalized[slug] = info

    return normalized


def save_cards_minimal(cards: Dict[str, dict]) -> None:
    """
    Save only:
      { slug: { "index": int, "cost": int or null } }
    """
    minimal: Dict[str, dict] = {}
    for slug, info in cards.items():
        idx = info.get("index")
        cost = info.get("cost")
        minimal[slug] = {
            "index": int(idx) if idx is not None else None,
            "cost": int(cost) if cost is not None else None,
        }

    with CARDS_JSON.open("w", encoding="utf-8") as f:
        json.dump(minimal, f, indent=2, ensure_ascii=False)
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
    Collect all <a> tags that have href starting with /card/detail/.
    """
    links: List[Tag] = []

    for sib in start_heading.next_siblings:
        if isinstance(sib, NavigableString):
            continue
        if isinstance(sib, Tag):
            if is_heading(sib):
                break  # end of this section

            for a in sib.find_all("a", href=True):
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


def card_cost_from_link(a_tag: Tag) -> Optional[int]:
    """
    Try to get elixir cost from the <div data-elixir="X"> inside the link.
    """
    div = a_tag.find("div")
    if not div:
        return None
    val = div.get("data-elixir")
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


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
    return " ".join(part.capitalize() for part in slug.split("-"))


def try_find_self_cost(soup: BeautifulSoup, card_name: str) -> Optional[int]:
    """
    Try to find the elixir cost for the card on its own detail page.
    Strategy: look for any element with data-elixir whose descendant <img alt="">
    matches the card name (case-insensitive).
    """
    name_lower = card_name.lower()
    for el in soup.find_all(attrs={"data-elixir": True}):
        val = el.get("data-elixir")
        try:
            cost = int(val)
        except (TypeError, ValueError):
            continue

        img = el.find("img", alt=True)
        if img and img["alt"].strip().lower() == name_lower:
            return cost

    return None


def build_matrices(session: requests.Session, cards: Dict[str, dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    cards: {slug: {...}}
    returns:
      counter_matrix, synergy_matrix
    also fills cards[slug]["index"] and cards[slug]["cost"] (if found).
    """
    slugs = sorted(cards.keys())
    n = len(slugs)

    # Assign indices
    slug_to_index: Dict[str, int] = {}
    for idx, slug in enumerate(slugs):
        slug_to_index[slug] = idx
        entry = cards.setdefault(slug, {})
        entry["index"] = idx
        # don't override cost here; we will fill/keep later

    counter_matrix = np.zeros((n, n), dtype=np.float32)
    synergy_matrix = np.zeros((n, n), dtype=np.float32)

    # Process each card detail page
    for slug in slugs:
        i = slug_to_index[slug]
        entry = cards[slug]
        card_name = entry.get("name") or derive_card_name_from_slug(slug)
        # keep card_name local only; we don't save it

        soup = fetch_card_detail(session, slug)
        time.sleep(REQUEST_DELAY_SEC)

        # Try to detect self cost from its own page
        if entry.get("cost") is None:
            self_cost = try_find_self_cost(soup, card_name)
            if self_cost is not None:
                entry["cost"] = self_cost

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
                    # This target card is not in cards.json – ignore
                    continue

                j = slug_to_index[target_slug]
                t_entry = cards.setdefault(target_slug, {})

                # Try to set cost for the target card from this link
                if t_entry.get("cost") is None:
                    c = card_cost_from_link(a)
                    if c is not None:
                        t_entry["cost"] = c

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

    # Save updated cards.json with only index + cost
    save_cards_minimal(cards)


if __name__ == "__main__":
    main()
