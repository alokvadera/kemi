import pytest

from kemi.adapters.embedding.custom import CustomEmbedAdapter


def test_embed_returns_vectors() -> None:
    embed_fn = lambda texts: [[0.1] * 32 for _ in texts]
    adapter = CustomEmbedAdapter(embed_fn=embed_fn, dim=32)

    texts = ["hello", "world"]
    result = adapter.embed(texts)

    assert len(result) == 2
    assert len(result[0]) == 32
    assert len(result[1]) == 32


def test_embed_single_delegates() -> None:
    embed_fn = lambda texts: [[0.1] * 32 for _ in texts]
    adapter = CustomEmbedAdapter(embed_fn=embed_fn, dim=32)

    result = adapter.embed_single("hello")

    assert len(result) == 32


def test_dimension_returns_dim() -> None:
    embed_fn = lambda texts: [[0.1] * 64 for _ in texts]
    adapter = CustomEmbedAdapter(embed_fn=embed_fn, dim=64)

    assert adapter.dimension() == 64


def test_embed_single_returns_single_vector() -> None:
    embed_fn = lambda texts: [[float(len(t))] for t in texts]
    adapter = CustomEmbedAdapter(embed_fn=embed_fn, dim=1)

    result = adapter.embed_single("hello")
    assert len(result) == 1
    assert result[0] == 5.0
