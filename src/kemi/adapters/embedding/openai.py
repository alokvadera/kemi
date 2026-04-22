import os
from kemi.adapters.base import EmbeddingAdapter


class OpenAIEmbedAdapter(EmbeddingAdapter):
    """Embedding adapter using OpenAI's embedding API.

    Default model: text-embedding-3-small
    Default dimension: 1536
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSION = 1536

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model_name = model_name or self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            except ImportError as e:
                raise ImportError("openai not installed. Run: pip install kemi[openai]") from e
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()
        response = client.embeddings.create(model=self._model_name, input=texts)

        return [item.embedding for item in response.data]

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def dimension(self) -> int:
        return self.DEFAULT_DIMENSION
