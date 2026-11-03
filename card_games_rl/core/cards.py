"""Cards are plain ints 0..51: ``suit * 13 + (rank - 2)``.

Ranks run 2..14 (Ace high = 14); suits are clubs, diamonds, hearts, spades.
"""
from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np

SUITS = ("C", "D", "H", "S")
SUIT_SYMBOLS = ("♣", "♦", "♥", "♠")
RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}
CLUBS, DIAMONDS, HEARTS, SPADES = range(4)
NUM_CARDS = 52


def make_card(rank: int, suit: int) -> int:
    return suit * 13 + (rank - 2)


def rank_of(card: int) -> int:
    return card % 13 + 2


def suit_of(card: int) -> int:
    return card // 13


def is_red(card: int) -> bool:
    return suit_of(card) in (DIAMONDS, HEARTS)


def rank_label(rank: int) -> str:
    return RANK_NAMES.get(rank, str(rank))


def label(card: int) -> str:
    """Short text label, e.g. ``"QS"`` or ``"10H"``."""
    return f"{rank_label(rank_of(card))}{SUITS[suit_of(card)]}"


def card_json(card: int) -> dict:
    return {"id": card, "rank": rank_label(rank_of(card)), "suit": SUITS[suit_of(card)], "label": label(card)}


def parse(text: str) -> int:
    """Inverse of :func:`label` (accepts ``"T"`` for ten), handy in tests."""
    text = text.strip().upper()
    rank_text, suit_text = text[:-1], text[-1]
    names = {v: k for k, v in RANK_NAMES.items()} | {"T": 10}
    rank = names[rank_text] if rank_text in names else int(rank_text)
    return make_card(rank, SUITS.index(suit_text))


def cards(text: str) -> list[int]:
    return [parse(token) for token in text.split()]


def new_deck(rng: random.Random | None = None, decks: int = 1) -> list[int]:
    deck = list(range(NUM_CARDS)) * decks
    if rng is not None:
        rng.shuffle(deck)
    return deck


def one_hot(card_ids: Iterable[int], size: int = NUM_CARDS) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    for card in card_ids:
        vector[card] = 1.0
    return vector


def one_hot_index(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vector[index] = 1.0
    return vector
