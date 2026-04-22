import json
import os
import tempfile

import pytest

from kemi.adapters.storage.json import JSONStorageAdapter
from kemi import Memory
from kemi.adapters.embedding.custom import CustomEmbedAdapter


def custom_embed(texts: list[str]) -> list[list[float]]:
    """Simple deterministic embedding for testing."""
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
def memory_with_data(tmp_path):
    """Create a memory instance with some test data using JSON storage."""
    db_path = str(tmp_path / "test.json")
    store = JSONStorageAdapter(path=db_path)
    embed = CustomEmbedAdapter(embed_fn=custom_embed, dim=64)
    memory = Memory(store=store, embed=embed)

    memory.remember("alice", "I love Python programming")
    memory.remember("alice", "I live in Mumbai city")
    memory.remember("bob", "Dark mode is preferred")

    return memory, db_path


class TestExport:
    def test_export_creates_valid_json_file(self, memory_with_data):
        memory, db_path = memory_with_data
        export_path = db_path.replace("test.json", "export.json")

        count = memory.export(export_path)

        assert os.path.exists(export_path)
        with open(export_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_export_returns_correct_count(self, memory_with_data):
        memory, db_path = memory_with_data
        export_path = db_path.replace("test.json", "export.json")

        count = memory.export(export_path)

        assert count == 3

    def test_export_contains_all_fields(self, memory_with_data):
        memory, db_path = memory_with_data
        export_path = db_path.replace("test.json", "export.json")

        memory.export(export_path)

        with open(export_path, "r") as f:
            data = json.load(f)

        assert len(data) == 3
        for mem in data:
            assert "memory_id" in mem
            assert "user_id" in mem
            assert "content" in mem
            assert "importance" in mem
            assert "lifecycle_state" in mem
            assert "created_at" in mem


class TestImport:
    def test_import_from_loads_memories_correctly(self, memory_with_data):
        memory, db_path = memory_with_data
        export_path = db_path.replace("test.json", "export.json")

        memory.export(export_path)

        new_db = db_path.replace("test.json", "new.json")
        new_store = JSONStorageAdapter(path=new_db)
        new_memory = Memory(
            store=new_store, embed=CustomEmbedAdapter(embed_fn=custom_embed, dim=64)
        )

        imported_count = new_memory.import_from(export_path)

        assert imported_count == 3
        results = new_memory.recall("alice", "anything")
        assert len(results) == 2
        results = new_memory.recall("bob", "anything")
        assert len(results) == 1

    def test_import_from_skips_duplicate_memory_ids(self, memory_with_data):
        memory, db_path = memory_with_data
        export_path = db_path.replace("test.json", "export.json")

        memory.export(export_path)

        imported_count = memory.import_from(export_path)

        assert imported_count == 0

    def test_import_from_returns_correct_count(self, memory_with_data):
        memory, db_path = memory_with_data
        export_path = db_path.replace("test.json", "export.json")

        memory.export(export_path)

        new_db = db_path.replace("test.json", "new.json")
        new_store = JSONStorageAdapter(path=new_db)
        new_memory = Memory(
            store=new_store, embed=CustomEmbedAdapter(embed_fn=custom_embed, dim=64)
        )

        imported_count = new_memory.import_from(export_path)

        assert imported_count == 3


class TestRoundTrip:
    def test_export_then_import_gives_same_memories(self, memory_with_data):
        memory, db_path = memory_with_data
        export_path = db_path.replace("test.json", "export.json")

        alice_results_before = memory.recall("alice", "anything")
        bob_results_before = memory.recall("bob", "anything")

        memory.export(export_path)

        new_db = db_path.replace("test.json", "new.json")
        new_store = JSONStorageAdapter(path=new_db)
        new_memory = Memory(
            store=new_store, embed=CustomEmbedAdapter(embed_fn=custom_embed, dim=64)
        )

        new_memory.import_from(export_path)

        alice_results_after = new_memory.recall("alice", "anything")
        bob_results_after = new_memory.recall("bob", "anything")

        assert len(alice_results_before) == len(alice_results_after)
        assert len(bob_results_before) == len(bob_results_after)


class TestAsyncExportImport:
    @pytest.mark.asyncio
    async def test_aexport_works(self, memory_with_data):
        memory, db_path = memory_with_data
        export_path = db_path.replace("test.json", "export_async.json")

        count = await memory.aexport(export_path)

        assert count == 3
        assert os.path.exists(export_path)

    @pytest.mark.asyncio
    async def test_aimport_from_works(self, memory_with_data):
        memory, db_path = memory_with_data
        export_path = db_path.replace("test.json", "export_async2.json")

        await memory.aexport(export_path)

        new_db = db_path.replace("test.json", "new.json")
        new_store = JSONStorageAdapter(path=new_db)
        new_memory = Memory(
            store=new_store, embed=CustomEmbedAdapter(embed_fn=custom_embed, dim=64)
        )

        imported_count = await new_memory.aimport_from(export_path)

        assert imported_count == 3
