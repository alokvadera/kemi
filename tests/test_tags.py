import pytest


def test_remember_with_tags(mock_memory) -> None:
    result = mock_memory.remember("user123", "I am vegetarian", tags=["food", "diet"])
    assert isinstance(result, str)

    mem = mock_memory._store.get(result)
    assert mem.tags == ["food", "diet"]


def test_remember_with_single_tag(mock_memory) -> None:
    result = mock_memory.remember("user123", "I live in Mumbai", tags=["location"])
    assert isinstance(result, str)

    mem = mock_memory._store.get(result)
    assert mem.tags == ["location"]


def test_remember_without_tags(mock_memory) -> None:
    result = mock_memory.remember("user123", "I am vegetarian")
    assert isinstance(result, str)

    mem = mock_memory._store.get(result)
    assert mem.tags == []


def test_remember_many_with_tags(mock_memory) -> None:
    contents = ["I love pizza and pasta", "I live in Mumbai"]
    ids = mock_memory.remember_many("user123", contents, tags=["food", "italian"])

    assert len(ids) == 2
    for mem_id in ids:
        mem = mock_memory._store.get(mem_id)
        assert mem.tags == ["food", "italian"]


def test_remember_many_without_tags(mock_memory) -> None:
    contents = ["I love pizza", "I love pasta"]
    ids = mock_memory.remember_many("user123", contents)

    assert len(ids) == 2
    for mem_id in ids:
        mem = mock_memory._store.get(mem_id)
        assert mem.tags == []


def test_recall_by_tag(mock_memory) -> None:
    mock_memory.remember("user123", "Pizza is my favorite food", tags=["food", "italian"])
    mock_memory.remember("user123", "I live in Australia", tags=["location", "australia"])
    mock_memory.remember("user123", "I eat pasta weekly", tags=["food", "italian"])

    results = mock_memory.recall_by_tag("user123", "food")
    assert len(results) >= 1

    for mem in results:
        assert "food" in mem.tags


def test_recall_by_tag_single_result(mock_memory) -> None:
    mock_memory.remember("user123", "I live in Mumbai", tags=["location", "india"])

    results = mock_memory.recall_by_tag("user123", "location")
    assert len(results) == 1
    assert "location" in results[0].tags


def test_recall_by_tag_no_results(mock_memory) -> None:
    mock_memory.remember("user123", "I love pizza", tags=["food"])

    results = mock_memory.recall_by_tag("user123", "nonexistent")
    assert results == []


def test_recall_by_tag_different_user(mock_memory) -> None:
    mock_memory.remember("user123", "I love pizza", tags=["food"])
    mock_memory.remember("user456", "I love pasta", tags=["food"])

    results = mock_memory.recall_by_tag("user123", "food")
    assert len(results) == 1
    assert results[0].user_id == "user123"


def test_recall_by_tag_with_lifecycle_filter(mock_memory) -> None:
    from kemi.models import LifecycleState

    mock_memory.remember("user123", "I love pizza", tags=["food"])

    results = mock_memory.recall_by_tag("user123", "food", lifecycle_filter=[LifecycleState.ACTIVE])
    assert len(results) == 1


def test_recall_by_tag_empty_tag(mock_memory) -> None:
    with pytest.raises(ValueError, match="tag cannot be empty"):
        mock_memory.recall_by_tag("user123", "")


def test_recall_by_tag_empty_user_id(mock_memory) -> None:
    with pytest.raises(ValueError, match="user_id cannot be empty"):
        mock_memory.recall_by_tag("", "food")


@pytest.mark.asyncio
async def test_aremember_with_tags(mock_memory) -> None:
    result = await mock_memory.aremember("user123", "I am vegetarian", tags=["food", "diet"])
    assert isinstance(result, str)

    mem = mock_memory._store.get(result)
    assert mem.tags == ["food", "diet"]


@pytest.mark.asyncio
async def test_aremember_many_with_tags(mock_memory) -> None:
    contents = ["I love pizza and pasta", "I live in Mumbai"]
    ids = await mock_memory.aremember_many("user123", contents, tags=["food"])

    assert len(ids) == 2
    for mem_id in ids:
        mem = mock_memory._store.get(mem_id)
        assert mem.tags == ["food"]


@pytest.mark.asyncio
async def test_arecall_by_tag(mock_memory) -> None:
    await mock_memory.aremember("user123", "Pizza is delicious food", tags=["food", "italian"])
    await mock_memory.aremember("user123", "I live in Mumbai", tags=["location"])

    results = await mock_memory.arecall_by_tag("user123", "food")
    assert len(results) == 1
    assert "food" in results[0].tags


def test_export_includes_tags(mock_memory, tmp_path) -> None:
    mock_memory.remember("user123", "Pizza is my favorite food", tags=["food", "italian"])
    mock_memory.remember("user123", "I live in Mumbai city", tags=["location"])

    file_path = tmp_path / "export.json"
    count = mock_memory.export(str(file_path))

    assert count == 2

    import json

    with open(file_path) as f:
        data = json.load(f)

    tags = [m["tags"] for m in data]
    assert ["food", "italian"] in tags
    assert ["location"] in tags


def test_import_includes_tags(mock_memory, tmp_path) -> None:
    import json

    file_path = tmp_path / "import.json"
    data = [
        {
            "memory_id": "mem1",
            "user_id": "user123",
            "content": "I love pizza",
            "tags": ["food", "italian"],
            "source": "user_stated",
            "importance": 0.5,
            "lifecycle_state": "active",
            "created_at": "2024-01-01T00:00:00",
            "last_accessed_at": "2024-01-01T00:00:00",
            "metadata": {},
            "embedding_dim": 64,
            "embedding": [0.1] * 64,
            "score": 0.0,
        }
    ]
    with open(file_path, "w") as f:
        json.dump(data, f)

    count = mock_memory.import_from(str(file_path))
    assert count == 1

    mem = mock_memory._store.get("mem1")
    assert mem.tags == ["food", "italian"]


def test_tags_default_empty_list(mock_memory) -> None:
    result = mock_memory.remember("user123", "Some content")
    mem = mock_memory._store.get(result)
    assert mem.tags == []
