from typing import Callable

from kemi.adapters.base import EmbeddingAdapter


class CustomEmbedAdapter(EmbeddingAdapter):
    """Custom embedding adapter that delegates to a user-provided function.

    Zero external dependencies.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        dim: int,
    ):
        self._embed_fn = embed_fn
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_fn(texts)

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def dimension(self) -> int:
        return self._dim
