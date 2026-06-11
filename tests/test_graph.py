"""Tests for src/kemi/graph.py — entity and relation extraction."""

from unittest.mock import MagicMock

from kemi.memory.model import LifecycleState
from kemi.nlp.graph import (
    _find_nearest_entity,
    _guess_entity_label,
    build_memory_graph,
    extract_entities,
    extract_relations,
)


class TestExtractEntities:
    def test_empty_string(self) -> None:
        assert extract_entities("") == []

    def test_no_entities(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        result = extract_entities(text)
        # No capitalized multi-word phrases, no emails, no URLs
        assert result == []

    def test_email_extraction(self) -> None:
        text = "Contact us at support@example.com for help."
        result = extract_entities(text)
        emails = [e for e in result if e["label"] == "EMAIL"]
        assert len(emails) == 1
        assert emails[0]["text"] == "support@example.com"

    def test_url_extraction(self) -> None:
        text = "Visit https://example.com or www.example.org for more."
        result = extract_entities(text)
        urls = [e for e in result if e["label"] == "URL"]
        assert len(urls) == 2

    def test_capitalized_phrase(self) -> None:
        text = "Alice Smith visited New York City."
        result = extract_entities(text)
        texts = [e["text"] for e in result]
        assert "Alice Smith" in texts
        assert "New York City" in texts

    def test_deduplication(self) -> None:
        text = "Alice Smith met Alice Smith again."
        result = extract_entities(text)
        alice = [e for e in result if e["text"] == "Alice Smith"]
        assert len(alice) == 1

    def test_organization_suffix(self) -> None:
        text = "Acme Inc is hiring. TechCorp Ltd announced profits."
        result = extract_entities(text)
        orgs = [e for e in result if e["label"] == "ORGANIZATION"]
        assert len(orgs) == 2

    def test_location_suffix(self) -> None:
        text = "We visited Springfield City and Colorado River."
        result = extract_entities(text)
        locs = [e for e in result if e["label"] == "LOCATION"]
        assert len(locs) == 2

    def test_person_name(self) -> None:
        text = "John visited the museum."
        result = extract_entities(text)
        persons = [e for e in result if e["label"] == "PERSON"]
        assert len(persons) == 1
        assert persons[0]["text"] == "John"

    def test_positions_are_correct(self) -> None:
        text = "Alice is here."
        result = extract_entities(text)
        assert len(result) == 1
        ent = result[0]
        assert ent["start"] == 0
        assert ent["end"] == 5

    def test_multiple_entity_types(self) -> None:
        text = "alice@acme.com"
        result = extract_entities(text)
        labels = {e["label"] for e in result}
        assert labels == {"EMAIL"}


class TestGuessEntityLabel:
    def test_person_from_common_name(self) -> None:
        assert _guess_entity_label("John") == "PERSON"
        assert _guess_entity_label("Alice") == "PERSON"

    def test_organization_from_suffix(self) -> None:
        assert _guess_entity_label("Acme Inc") == "ORGANIZATION"
        assert _guess_entity_label("TechCorp Ltd") == "ORGANIZATION"

    def test_location_from_suffix(self) -> None:
        assert _guess_entity_label("Springfield City") == "LOCATION"
        assert _guess_entity_label("Colorado River") == "LOCATION"

    def test_default_entity(self) -> None:
        assert _guess_entity_label("Something Unknown") == "ENTITY"


class TestExtractRelations:
    def test_empty_text(self) -> None:
        assert extract_relations("", []) == []

    def test_no_entities(self) -> None:
        text = "John loves Python."
        assert extract_relations(text, []) == []

    def test_relation_with_match(self) -> None:
        text = "John lives in London."
        entities = [
            {"text": "John", "start": 0, "end": 4, "label": "PERSON"},
            {"text": "London", "start": 14, "end": 20, "label": "LOCATION"},
        ]
        result = extract_relations(text, entities)
        assert len(result) == 1
        assert result[0]["subject"] == "John"
        assert result[0]["predicate"] == "LOCATED_AT"
        assert result[0]["object"] == "London"

    def test_multiple_relations(self) -> None:
        text = "Alice works at Google and likes Python."
        entities = [
            {"text": "Alice", "start": 0, "end": 5, "label": "PERSON"},
            {"text": "Google", "start": 16, "end": 22, "label": "ORGANIZATION"},
            {"text": "Python", "start": 34, "end": 40, "label": "ENTITY"},
        ]
        result = extract_relations(text, entities)
        predicates = {r["predicate"] for r in result}
        assert "WORKS_AT" in predicates
        assert "PREFERS" in predicates

    def test_no_relation_keyword(self) -> None:
        text = "Alice and Bob went to the store."
        entities = [
            {"text": "Alice", "start": 0, "end": 5, "label": "PERSON"},
            {"text": "Bob", "start": 10, "end": 13, "label": "PERSON"},
        ]
        result = extract_relations(text, entities)
        assert result == []


class TestFindNearestEntity:
    def test_empty_entities(self) -> None:
        assert _find_nearest_entity([], 10, before=True) is None

    def test_exact_match_before(self) -> None:
        entities = [
            {"text": "A", "start": 0, "end": 1},
            {"text": "B", "start": 5, "end": 6},
        ]
        result = _find_nearest_entity(entities, 5, before=True)
        assert result is not None
        assert result["text"] == "A"

    def test_exact_match_after(self) -> None:
        entities = [
            {"text": "A", "start": 0, "end": 1},
            {"text": "B", "start": 5, "end": 6},
        ]
        result = _find_nearest_entity(entities, 1, before=False)
        assert result is not None
        assert result["text"] == "B"

    def test_no_match_before(self) -> None:
        entities = [
            {"text": "B", "start": 5, "end": 6},
        ]
        result = _find_nearest_entity(entities, 3, before=True)
        assert result is None


class TestBuildMemoryGraph:
    def test_empty_store(self) -> None:
        mock_store = MagicMock()
        mock_store.get_all_by_user.return_value = []
        result = build_memory_graph(mock_store, "alice")
        assert result["entities"] == []
        assert result["relations"] == []

    def test_single_memory(self) -> None:
        mock_store = MagicMock()
        mem = MagicMock()
        mem.content = "Alice lives in London."
        mem.lifecycle_state = LifecycleState.ACTIVE
        mock_store.get_all_by_user.return_value = [mem]
        result = build_memory_graph(mock_store, "alice")
        assert len(result["entities"]) >= 1
        assert any(e["text"] == "Alice" for e in result["entities"])

    def test_filters_by_lifecycle(self) -> None:
        mock_store = MagicMock()
        active_mem = MagicMock()
        active_mem.content = "Alice likes Python."
        active_mem.lifecycle_state = LifecycleState.ACTIVE
        mock_store.get_all_by_user.return_value = [active_mem]
        result = build_memory_graph(mock_store, "alice")
        # Verify the store was called with the correct lifecycle filter
        call_kwargs = mock_store.get_all_by_user.call_args.kwargs
        assert LifecycleState.ACTIVE in call_kwargs["lifecycle_filter"]
        assert LifecycleState.DECAYING in call_kwargs["lifecycle_filter"]
        assert LifecycleState.DELETED not in call_kwargs["lifecycle_filter"]
        # Result should only contain entities from the returned memory
        texts = [e["text"] for e in result["entities"]]
        assert "Alice" in texts or "Python" in texts

    def test_namespace_passed_through(self) -> None:
        mock_store = MagicMock()
        mock_store.get_all_by_user.return_value = []
        build_memory_graph(mock_store, "alice", namespace="work")
        call_kwargs = mock_store.get_all_by_user.call_args.kwargs
        assert call_kwargs["namespace"] == "work"

    def test_deduplicates_entities(self) -> None:
        mock_store = MagicMock()
        mem1 = MagicMock()
        mem1.content = "Alice likes Python."
        mem1.lifecycle_state = LifecycleState.ACTIVE
        mem2 = MagicMock()
        mem2.content = "Alice likes Java."
        mem2.lifecycle_state = LifecycleState.ACTIVE
        mock_store.get_all_by_user.return_value = [mem1, mem2]
        result = build_memory_graph(mock_store, "alice")
        alice_count = sum(1 for e in result["entities"] if e["text"] == "Alice")
        assert alice_count == 1

    def test_deduplicates_relations(self) -> None:
        mock_store = MagicMock()
        mem1 = MagicMock()
        mem1.content = "Alice works at Google."
        mem1.lifecycle_state = LifecycleState.ACTIVE
        mem2 = MagicMock()
        mem2.content = "Alice works at Google."
        mem2.lifecycle_state = LifecycleState.ACTIVE
        mock_store.get_all_by_user.return_value = [mem1, mem2]
        result = build_memory_graph(mock_store, "alice")
        rels = [r for r in result["relations"] if r["predicate"] == "WORKS_AT"]
        assert len(rels) == 1
