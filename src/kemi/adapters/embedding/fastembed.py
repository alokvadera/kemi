from typing import Optional

from kemi.adapters.base import EmbeddingAdapter


class FastEmbedAdapter(EmbeddingAdapter):
    """Embedding adapter using fastembed BAAI/bge-small-en-v1.5 model.

    Model dimension: 384

    Lazy-loads the model on first embed() call to avoid import-time errors.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DIMENSION = 384

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or self.MODEL_NAME
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self._model_name)
            except ImportError as e:
                raise ImportError("fastembed not installed. Run: pip install kemi[local]") from e
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        return [emb.tolist() for emb in model.embed(texts)]

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def dimension(self) -> int:
        return self.DIMENSION
