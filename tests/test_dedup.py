"""Tests for src/kemi/dedup.py — memory deduplication and conflict detection."""

from __future__ import annotations

import math

from kemi.memory import dedup
from tests._helpers.factories import make_memory


def test_find_duplicates_above_threshold() -> None:
    new_mem = make_memory(
        memory_id="new",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
    )
    existing = [
        make_memory(
            memory_id="old",
            user_id="user",
            content="I am vegetarian",
            embedding=[1.0] * 64,
        )
    ]

    result = dedup.find_duplicates(new_mem, existing, threshold=0.85)
    assert len(result) == 1
    assert result[0].memory_id == "old"


def test_find_duplicates_below_threshold() -> None:
    new_mem = make_memory(
        memory_id="new",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
    )
    existing = [
        make_memory(
            memory_id="old",
            user_id="user",
            content="I live in NYC",
            embedding=[1.0 if i % 2 == 0 else -1.0 for i in range(64)],
        )
    ]

    result = dedup.find_duplicates(new_mem, existing, threshold=0.85)
    assert len(result) == 0


def test_find_conflicts_in_range() -> None:
    rad = 50 * math.pi / 180
    new_mem = make_memory(
        memory_id="new",
        user_id="user",
        content="I like running",
        embedding=[1.0, 0.0] * 32,
    )
    conflicting = [math.cos(rad), math.sin(rad)] * 32
    existing = [
        make_memory(
            memory_id="old",
            user_id="user",
            content="I hate running",
            embedding=conflicting,
        )
    ]

    result = dedup.find_conflicts(
        new_mem, existing, conflict_threshold=0.65, dedup_threshold=0.85
    )
    assert len(result) == 1


def test_find_duplicates_and_conflicts_no_overlap() -> None:
    new_mem = make_memory(
        memory_id="new",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
    )
    existing = [
        make_memory(
            memory_id="dup",
            user_id="user",
            content="I am vegetarian",
            embedding=[1.0] * 64,
        )
    ]

    duplicates = dedup.find_duplicates(new_mem, existing, threshold=0.85)
    conflicts = dedup.find_conflicts(
        new_mem, existing, conflict_threshold=0.65, dedup_threshold=0.85
    )

    duplicate_ids = {m.memory_id for m in duplicates}
    conflict_ids = {m.memory_id for m in conflicts}
    assert duplicate_ids.isdisjoint(conflict_ids)


def test_resolve_duplicate_preserves_memory_id() -> None:
    new_mem = make_memory(
        memory_id="new",
        user_id="user",
        content="I am vegetarian now",
        embedding=[1.0] * 64,
    )
    existing = make_memory(
        memory_id="old-id",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
    )

    resolved = dedup.resolve_duplicate(new_mem, existing)
    assert resolved.memory_id == "old-id"


def test_resolve_duplicate_updates_content() -> None:
    new_mem = make_memory(
        memory_id="new",
        user_id="user",
        content="I am vegetarian now",
        embedding=[1.0] * 64,
    )
    existing = make_memory(
        memory_id="old-id",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
    )

    resolved = dedup.resolve_duplicate(new_mem, existing)
    assert resolved.content == "I am vegetarian now"


def test_resolve_duplicate_no_mutation() -> None:
    new_mem = make_memory(
        memory_id="new",
        user_id="user",
        content="I am vegetarian now",
        embedding=[1.0] * 64,
    )
    existing = make_memory(
        memory_id="old-id",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
    )

    original_existing_content = existing.content
    original_new_content = new_mem.content

    dedup.resolve_duplicate(new_mem, existing)

    assert existing.content == original_existing_content
    assert new_mem.content == original_new_content


def test_find_duplicates_with_none_embedding() -> None:
    new_mem = make_memory(
        memory_id="new",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
    )
    existing = [
        make_memory(
            memory_id="old",
            user_id="user",
            content="I also am vegetarian",
            embedding=None,
        )
    ]

    result = dedup.find_duplicates(new_mem, existing, threshold=0.85)
    assert len(result) == 0


def test_find_conflicts_with_none_embedding() -> None:
    new_mem = make_memory(
        memory_id="new",
        user_id="user",
        content="I like running",
        embedding=[0.75] * 64,
    )
    existing = [
        make_memory(
            memory_id="old",
            user_id="user",
            content="I hate running",
            embedding=None,
        )
    ]

    result = dedup.find_conflicts(
        new_mem, existing, conflict_threshold=0.65, dedup_threshold=0.85
    )
    assert len(result) == 0


def test_has_sentiment_flip_detects_flip() -> None:
    result = dedup.has_sentiment_flip("I love cats", "I hate cats")
    assert result is True


def test_has_sentiment_flip_no_flip() -> None:
    result = dedup.has_sentiment_flip("I love cats", "I love dogs")
    assert result is False


def test_has_sentiment_flip_no_common_nouns() -> None:
    result = dedup.has_sentiment_flip("I love pizza", "I hate running")
    assert result is False


def test_has_sentiment_flip_negation_mismatch() -> None:
    result = dedup.has_sentiment_flip("I do not like rain", "I like rain")
    assert result is True


def test_has_sentiment_flip_same_text() -> None:
    result = dedup.has_sentiment_flip("I enjoy music", "I enjoy music")
    assert result is False
