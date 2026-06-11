"""Root conftest: re-exports fixtures from ``_helpers``.

The canonical mock classes and factory functions now live in
``tests/_helpers/``. This conftest is a thin re-exporter so existing
test files that import ``mock_memory`` / ``real_db_memory`` /
``mock_embedding`` / ``mock_storage`` keep working unchanged.
"""

from __future__ import annotations

import pytest

from kemi import Memory
from tests._helpers.factories import make_real_db_memory
from tests._helpers.mock_storage import MockEmbeddingAdapter, MockStorageAdapter

mock_embedding_factory = MockEmbeddingAdapter
mock_storage_factory = MockStorageAdapter


@pytest.fixture
def mock_embedding() -> type:
    return MockEmbeddingAdapter


@pytest.fixture
def mock_storage() -> type:
    return MockStorageAdapter


@pytest.fixture
def mock_memory(mock_embedding: type, mock_storage: type) -> Memory:
    return Memory(embed=mock_embedding(), store=mock_storage())


@pytest.fixture
def real_db_memory(tmp_path, mock_embedding: type) -> Memory:
    """Create a Memory instance backed by a real temporary SQLite database.

    This exercises the full CLI-to-storage pipeline including schema creation,
    SQL INSERT/SELECT operations, and storage adapter implementation.
    Uses the mock embedding adapter so no external embedding service is needed.
    """
    return make_real_db_memory(tmp_path, embed=mock_embedding())


__all__ = [
    "mock_embedding",
    "mock_storage",
    "mock_memory",
    "real_db_memory",
    "MockEmbeddingAdapter",
    "MockStorageAdapter",
]
