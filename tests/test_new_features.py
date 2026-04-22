import tempfile

import pytest

from kemi import Memory
from kemi.adapters.storage.json import JSONStorageAdapter
from kemi.adapters.embedding.custom import CustomEmbedAdapter


def custom_embed(texts: list[str]) -> list[list[float]]:
    import hashlib

    dim = 64
    vectors = []
    for text in texts:
        raw = hashlib.sha256(text.encode()).digest()
        expanded = raw * (dim // len(raw) + 1)
        vector = [b / 255.0 for b in expanded[:dim]]
        vectors.append(vector)
    return vectors


@pytest.fixture
def memory_with_users(tmp_path):
    """Create memory with multiple users."""
    db_path = str(tmp_path / "test.json")
    store = JSONStorageAdapter(path=db_path)
    embed = CustomEmbedAdapter(embed_fn=custom_embed, dim=64)
    memory = Memory(store=store, embed=embed)

    memory.remember("alice", "I love Python")
    memory.remember("alice", "I live in Mumbai")
    memory.remember("bob", "I prefer dark mode")
    memory.remember("charlie", "My name is Charlie")

    return memory, db_path


class TestListUsers:
    def test_list_users_returns_all_users(self, memory_with_users):
        memory, _ = memory_with_users

        users = memory.list_users()

        assert len(users) == 3
        assert "alice" in users
        assert "bob" in users
        assert "charlie" in users

    def test_list_users_no_duplicates(self, memory_with_users):
        memory, _ = memory_with_users

        users = memory.list_users()

        assert len(users) == len(set(users))

    @pytest.mark.asyncio
    async def test_alist_users(self, memory_with_users):
        memory, _ = memory_with_users

        users = await memory.alist_users()

        assert len(users) == 3


class TestUpdate:
    def test_update_content(self, memory_with_users):
        memory, _ = memory_with_users

        results = memory.recall("alice", "Python")
        memory_id = results[0].memory_id

        updated_id = memory.update(memory_id, content="I love JavaScript")

        assert updated_id == memory_id

        results = memory.recall("alice", "JavaScript")
        assert results[0].content == "I love JavaScript"

    def test_update_importance(self, memory_with_users):
        memory, _ = memory_with_users

        results = memory.recall("alice", "Python")
        memory_id = results[0].memory_id

        memory.update(memory_id, importance=0.9)

        results = memory.recall("alice", "Python")
        assert results[0].importance == 0.9

    def test_update_not_found(self, memory_with_users):
        memory, _ = memory_with_users

        with pytest.raises(ValueError, match="Memory not found"):
            memory.update("nonexistent-id", content="new content")

    @pytest.mark.asyncio
    async def test_aupdate(self, memory_with_users):
        memory, _ = memory_with_users

        results = memory.recall("alice", "Python")
        memory_id = results[0].memory_id

        await memory.aupdate(memory_id, content="I love Rust")

        results = memory.recall("alice", "Rust")
        assert results[0].content == "I love Rust"


class TestRecallSince:
    def test_recall_since_returns_results(self, memory_with_users):
        memory, _ = memory_with_users

        results = memory.recall_since("alice", "Python", hours=24)

        assert len(results) >= 1

    def test_recall_since_top_k(self, memory_with_users):
        memory, _ = memory_with_users

        results = memory.recall_since("alice", "anything", hours=24, top_k=1)

        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_arecall_since(self, memory_with_users):
        memory, _ = memory_with_users

        results = await memory.arecall_since("alice", "Python", hours=24)

        assert len(results) >= 1


class TestRememberMany:
    def test_remember_many_returns_list_of_ids(self, memory_with_users):
        memory, _ = memory_with_users

        ids = memory.remember_many(
            "dave", ["I love Python programming", "I live in Mumbai city", "I prefer dark mode"]
        )

        assert len(ids) == 3
        assert all(isinstance(i, str) for i in ids)

    def test_remember_many_stores_all(self, memory_with_users):
        memory, _ = memory_with_users

        memory.remember_many("eve", ["Python is great", "Mumbai is my city", "Dark mode preferred"])

        results = memory.recall("eve", "anything")
        assert len(results) >= 1

    def test_remember_many_with_importance(self, memory_with_users):
        memory, _ = memory_with_users

        memory.remember_many(
            "frank", ["One fact here", "Two fact here", "Three fact here"], importance=0.9
        )

        results = memory.recall("frank", "anything")
        assert all(r.importance == 0.9 for r in results)

    @pytest.mark.asyncio
    async def test_aremember_many(self, memory_with_users):
        memory, _ = memory_with_users

        ids = await memory.aremember_many("grace", ["Alpha fact", "Beta fact", "Gamma fact"])

        assert len(ids) == 3

    def test_remember_many_empty_list(self, memory_with_users):
        memory, _ = memory_with_users

        ids = memory.remember_many("empty_user", [])

        assert ids == []

    def test_remember_many_validates_content(self, memory_with_users):
        memory, _ = memory_with_users

        with pytest.raises(ValueError, match="content cannot be empty"):
            memory.remember_many("test_user", ["valid content", ""])
