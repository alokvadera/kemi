import pytest


def test_remember_returns_string_id(mock_memory) -> None:
    result = mock_memory.remember("user123", "I am vegetarian")
    assert isinstance(result, str)
    assert len(result) > 0


def test_remember_dedup(mock_memory) -> None:
    id1 = mock_memory.remember("user123", "I am vegetarian")
    id2 = mock_memory.remember("user123", "I am vegetarian")
    assert id1 == id2


def test_remember_different_content(mock_memory) -> None:
    id1 = mock_memory.remember("user123", "I am vegetarian")
    id2 = mock_memory.remember("user123", "I live in Mumbai")
    assert id1 != id2


def test_recall_returns_list(mock_memory) -> None:
    mock_memory.remember("user123", "I am vegetarian")
    result = mock_memory.recall("user123", "food preferences")
    assert isinstance(result, list)


def test_recall_empty_user(mock_memory) -> None:
    result = mock_memory.recall("newuser", "any query")
    assert result == []


def test_context_block_format(mock_memory) -> None:
    mock_memory.remember("user123", "I am vegetarian")
    mock_memory.remember("user123", "I live in Mumbai")
    result = mock_memory.context_block("user123", "user preferences")
    assert result.startswith("Relevant context from memory:")
    assert "- I am vegetarian" in result
    assert "- I live in Mumbai" in result


def test_context_block_empty(mock_memory) -> None:
    result = mock_memory.context_block("user123", "query")
    assert result == ""


def test_forget_by_id(mock_memory) -> None:
    mem_id = mock_memory.remember("user123", "I am vegetarian")
    result = mock_memory.forget("user123", mem_id)
    assert result == 1

    result = mock_memory.forget("user123", mem_id)
    assert result == 0


def test_forget_by_user(mock_memory) -> None:
    mock_memory.remember("user123", "I am vegetarian")
    mock_memory.remember("user123", "I live in Mumbai")
    result = mock_memory.forget("user123")
    assert result == 2
