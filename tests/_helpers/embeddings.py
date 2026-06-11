"""Deterministic embedding functions for tests.

Replaces scattered patterns like:
- ``[0.0] * 32`` (8 files)
- ``hashlib.sha256(text.encode()).digest()`` (conftest canonical)
- ``_embed_32`` (test_review_fixes.py)
- ``_embed_fn_32`` / ``_embed_fn_64`` (test_custom_embed.py)

Use the ``embed_*`` functions in place of inline ``[0.0] * N`` arrays when a
test only needs the *shape* of an embedding, and ``embed_single_*`` when a
test needs a single vector for one specific text.
"""

from __future__ import annotations

import hashlib


def _det(text: str, dim: int) -> list[float]:
    raw = hashlib.sha256(text.encode()).digest()
    expanded = raw * (dim // len(raw) + 1)
    return [b / 255.0 for b in expanded[:dim]]


def embed_32(texts: list[str]) -> list[list[float]]:
    """Deterministic 32-dim embedding for a batch of texts."""
    return [_det(t, 32) for t in texts]


def embed_64(texts: list[str]) -> list[list[float]]:
    """Deterministic 64-dim embedding for a batch of texts."""
    return [_det(t, 64) for t in texts]


def embed_single_32(text: str) -> list[float]:
    """Deterministic 32-dim embedding for a single text."""
    return _det(text, 32)


def embed_single_64(text: str) -> list[float]:
    """Deterministic 64-dim embedding for a single text."""
    return _det(text, 64)


def constant_vector(value: float, dim: int) -> list[float]:
    """Return a constant-valued vector of the given dim.

    Useful when a test only cares that two vectors are equal/different,
    not their actual values.
    """
    return [value] * dim
