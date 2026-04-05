from kemi.adapters.embedding.custom import CustomEmbedAdapter


def _embed_fn_32(texts):
    return [[0.1] * 32 for _ in texts]


def _embed_fn_64(texts):
    return [[0.1] * 64 for _ in texts]


def _embed_fn_len(texts):
    return [[float(len(t))] for t in texts]


def test_embed_returns_vectors() -> None:
    adapter = CustomEmbedAdapter(embed_fn=_embed_fn_32, dim=32)

    texts = ["hello", "world"]
    result = adapter.embed(texts)

    assert len(result) == 2
    assert len(result[0]) == 32
    assert len(result[1]) == 32


def test_embed_single_delegates() -> None:
    adapter = CustomEmbedAdapter(embed_fn=_embed_fn_32, dim=32)

    result = adapter.embed_single("hello")

    assert len(result) == 32


def test_dimension_returns_dim() -> None:
    adapter = CustomEmbedAdapter(embed_fn=_embed_fn_64, dim=64)

    assert adapter.dimension() == 64


def test_embed_single_returns_single_vector() -> None:
    adapter = CustomEmbedAdapter(embed_fn=_embed_fn_len, dim=1)

    result = adapter.embed_single("hello")
    assert len(result) == 1
    assert result[0] == 5.0
