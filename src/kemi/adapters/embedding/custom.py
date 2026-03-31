from typing import Callable, List

from kemi.adapters.base import EmbeddingAdapter


class CustomEmbedAdapter(EmbeddingAdapter):
    """Custom embedding adapter that delegates to a user-provided function.

    Zero external dependencies.
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], List[List[float]]],
        dim: int,
    ):
        self._embed_fn = embed_fn
        self._dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._embed_fn(texts)

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def dimension(self) -> int:
        return self._dim
