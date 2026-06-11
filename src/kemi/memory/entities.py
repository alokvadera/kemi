"""Entity extraction for entity-aware retrieval.

Provides pluggable entity linkers that extract normalized entity strings
from text. The default :class:`RegexEntityLinker` uses regex heuristics for
names, dates, emails, and URLs with zero external dependencies.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from kemi.exceptions import ConfigurationError


class EntityLinker(ABC):
    """Abstract interface for extracting entities from text.

    Implementations should return a set of **normalized** entity strings
    (e.g. lower-cased, stripped) so that overlap comparisons are case-insensitive.
    """

    @abstractmethod
    def extract(self, text: str) -> set[str]:
        """Extract entities from *text*.

        Args:
            text: Input string.

        Returns:
            Set of normalized entity strings.
        """
        pass

    def extract_batch(self, texts: list[str]) -> list[set[str]]:
        """Extract entities from multiple texts in a single batch.

        The default implementation loops over ``extract`` one-by-one.
        Subclasses that can batch more efficiently (e.g. spaCy pipelines
        with ``nlp.pipe``) SHOULD override this.

        Args:
            texts: List of input strings.

        Returns:
            List of entity sets, one per input text.
        """
        return [self.extract(t) for t in texts]


class NoopEntityLinker(EntityLinker):
    """No-op entity linker that returns an empty set.

    Used when entity-aware retrieval is disabled.
    """

    def extract(self, text: str) -> set[str]:
        return set()


class RegexEntityLinker(EntityLinker):
    """Regex-based entity linker.

    Extracts:
    - Capitalized phrases (names, places, organisations)
    - Email addresses
    - URLs
    - ISO-style dates (YYYY-MM-DD) and relaxed dates (Month DD, YYYY)

    All entities are normalised to lower-case.
    """

    _DATE_PATTERNS = [
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),          # 2024-06-05
        re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),   # 06/05/2024
        re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE),  # noqa: E501
    ]

    _EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    _URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+")
    _NAME_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")

    def extract(self, text: str) -> set[str]:
        entities: set[str] = set()

        # Names / capitalised phrases
        for match in self._NAME_PATTERN.finditer(text):
            entities.add(match.group().lower())

        # Dates
        for pattern in self._DATE_PATTERNS:
            for match in pattern.finditer(text):
                entities.add(match.group().lower())

        # Emails
        for match in self._EMAIL_PATTERN.finditer(text):
            entities.add(match.group().lower())

        # URLs
        for match in self._URL_PATTERN.finditer(text):
            entities.add(match.group().lower())

        return entities


class SpacyEntityLinker(EntityLinker):
    """spaCy NER-based entity linker.

    Uses spaCy’s named-entity recognition pipeline for accurate extraction
    of people, organisations, locations, dates, products, etc.

    Requires ``spacy`` and a language model (e.g. ``en_core_web_sm``) to be
    installed:

    .. code-block:: bash

        pip install spacy
        python -m spacy download en_core_web_sm

    All extracted entities are normalised to lower-case.

    Args:
        model: spaCy model name (default ``en_core_web_sm``).
        allowed_labels: Set of spaCy NER labels to keep.  If ``None``,
            a sensible default set is used.
            See https://spacy.io/usage/linguistic-features#named-entities
    """

    _DEFAULT_LABELS: set[str] = {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "DATE",
        "EVENT",
        "PRODUCT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "FAC",
        "NORP",
    }

    def __init__(
        self,
        model: str = "en_core_web_sm",
        allowed_labels: set[str] | None = None,
    ) -> None:
        try:
            import spacy
        except ImportError as exc:
            raise ConfigurationError(
                "spaCy is required for SpacyEntityLinker. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            ) from exc

        self._nlp: Any = spacy.load(model)
        self._allowed_labels: set[str] = allowed_labels if allowed_labels is not None else self._DEFAULT_LABELS  # noqa: E501

    def extract(self, text: str) -> set[str]:
        doc = self._nlp(text)
        entities: set[str] = set()
        for ent in doc.ents:
            if ent.label_ in self._allowed_labels:
                entities.add(ent.text.lower())
        return entities

    def extract_batch(self, texts: list[str]) -> list[set[str]]:
        """Batch entity extraction using ``spacy.pipe`` for efficiency."""
        results: list[set[str]] = []
        for doc in self._nlp.pipe(texts):
            entities: set[str] = set()
            for ent in doc.ents:
                if ent.label_ in self._allowed_labels:
                    entities.add(ent.text.lower())
            results.append(entities)
        return results
