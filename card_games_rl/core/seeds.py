"""Disjoint deal seeds for training, validation and test.

Validation seeds start at 1,000,000 and test seeds at 2,000,000 (each split
uses at most 1,000,000 deals). Training never draws a seed from that reserved
range, so held-out deals are never seen during training.
"""
from __future__ import annotations

import numpy as np

VALIDATION_SEEDS = 1_000_000
TEST_SEEDS = 2_000_000
RESERVED_END = 3_000_000
_MAX_SEED = 2**31 - 1


def training_seed(rng: np.random.Generator) -> int:
    """Uniform seed from [0, 2^31) minus the reserved validation/test range."""
    seed = int(rng.integers(_MAX_SEED - (RESERVED_END - VALIDATION_SEEDS)))
    return seed + (RESERVED_END - VALIDATION_SEEDS) if seed >= VALIDATION_SEEDS else seed


def is_reserved(seed: int) -> bool:
    return VALIDATION_SEEDS <= seed < RESERVED_END
