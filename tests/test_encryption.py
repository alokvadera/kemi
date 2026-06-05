"""Tests for at-rest encryption in storage adapters."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from kemi.encryption import (
    EncryptionConfig,
    FieldEncryptor,
    FernetEncryptor,
    generate_key,
    is_cryptography_available,
    is_sqlcipher_available,
    load_key_from_file,
)
from kemi.models import LifecycleState, MemoryObject, MemorySource, MemoryType


class TestFernetEncryptor:
    """Tests for Fernet field-level encryption."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypt and decrypt returns original data."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        key = generate_key()
        encryptor = FernetEncryptor(key)

        original = "Hello, this is a secret message!"
        encrypted = encryptor.encrypt(original)
        decrypted = encryptor.decrypt_str(encrypted)

        assert decrypted == original
        assert encrypted != original

    def test_encrypt_bytes(self):
        """Encrypt handles bytes input."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        key = generate_key()
        encryptor = FernetEncryptor(key)

        original = b"binary data with \\x00 bytes"
        encrypted = encryptor.encrypt(original)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == original

    def test_encrypted_output_is_different_each_time(self):
        """Fernet encryption is non-deterministic (random IV)."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        key = generate_key()
        encryptor = FernetEncryptor(key)

        text = "Same message twice"
        e1 = encryptor.encrypt(text)
        e2 = encryptor.encrypt(text)

        # Different ciphertexts due to random IV
        assert e1 != e2
        # But both decrypt to same original
        assert encryptor.decrypt_str(e1) == text
        assert encryptor.decrypt_str(e2) == text


class TestFieldEncryptor:
    """Tests for field-level encryption of memory rows."""

    def test_encrypt_memory_row_content_and_metadata(self):
        """FieldEncryptor encrypts content and metadata fields."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        config = EncryptionConfig(enabled=True, key=generate_key())
        encryptor = FieldEncryptor(config)

        row = {
            "memory_id": "mem-123",
            "user_id": "user1",
            "content": "My secret content",
            "metadata": {"secret": True, "value": 42},
            "lifecycle_state": "active",
        }

        encrypted = encryptor.encrypt_memory_row(row)

        # content should be encrypted dict with sentinel
        assert encrypted["content"] != "My secret content"
        assert encrypted["content"]["encrypted"] is True
        assert "data" in encrypted["content"]

        # metadata should be encrypted
        assert encrypted["metadata"] != {"secret": True, "value": 42}
        assert encrypted["metadata"]["encrypted"] is True

        # user_id should NOT be encrypted by default
        assert encrypted["user_id"] == "user1"
        # memory_id should NOT be encrypted
        assert encrypted["memory_id"] == "mem-123"

    def test_decrypt_memory_row_roundtrip(self):
        """Decrypting encrypted row returns original values."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        config = EncryptionConfig(enabled=True, key=generate_key())
        encryptor = FieldEncryptor(config)

        original_row = {
            "memory_id": "mem-456",
            "user_id": "alice",
            "content": "Confidential data here",
            "metadata": {"nested": {"deep": [1, 2, 3]}},
            "lifecycle_state": "active",
            "importance": 0.8,
        }

        encrypted = encryptor.encrypt_memory_row(original_row)
        decrypted = encryptor.decrypt_memory_row(encrypted)

        assert decrypted["content"] == "Confidential data here"
        assert decrypted["metadata"] == {"nested": {"deep": [1, 2, 3]}}
        assert decrypted["user_id"] == "alice"
        assert decrypted["memory_id"] == "mem-456"

    def test_disabled_encryption_passes_through(self):
        """When encryption is disabled, no transformation occurs."""
        config = EncryptionConfig(enabled=False)
        encryptor = FieldEncryptor(config)

        row = {"content": "plain text", "metadata": {"key": "value"}}
        result = encryptor.encrypt_memory_row(row)

        assert result["content"] == "plain text"
        assert result["metadata"] == {"key": "value"}

        decrypted = encryptor.decrypt_memory_row(row)
        assert decrypted == row

    def test_encrypt_user_id_option(self):
        """Optional user_id encryption works."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        config = EncryptionConfig(
            enabled=True,
            key=generate_key(),
            encrypt_user_id=True,
        )
        encryptor = FieldEncryptor(config)

        row = {"user_id": "secret-user", "content": "data"}
        encrypted = encryptor.encrypt_memory_row(row)

        assert encrypted["user_id"]["encrypted"] is True
        assert encrypted["user_id"]["data"] != "secret-user"

        decrypted = encryptor.decrypt_memory_row(encrypted)
        assert decrypted["user_id"] == "secret-user"

    def test_null_fields_handled(self):
        """None values in fields are preserved as None."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        config = EncryptionConfig(enabled=True, key=generate_key())
        encryptor = FieldEncryptor(config)

        row = {"content": None, "metadata": None, "user_id": "alice"}
        result = encryptor.encrypt_memory_row(row)

        assert result["content"] is None
        assert result["metadata"] is None


class TestEncryptionConfig:
    """Tests for EncryptionConfig loading and key management."""

    def test_from_key_file_loads_key(self):
        """from_key_file correctly loads key from file."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".key", delete=False) as f:
            f.write("test-encryption-key-12345678901234567890")
            f.flush()
            key_path = f.name

        try:
            config = EncryptionConfig.from_key_file(key_path, key_id="test-key")
            assert config.key == "test-encryption-key-12345678901234567890"
            assert config.key_id == "test-key"
            assert config.enabled is True
        finally:
            os.unlink(key_path)

    def test_from_key_file_raises_on_missing_file(self):
        """from_key_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            EncryptionConfig.from_key_file("/nonexistent/path/to/key")

    def test_key_property_raises_when_no_key(self):
        """key property raises ValueError when no key configured."""
        config = EncryptionConfig(enabled=True, key="", key_file=None)
        with pytest.raises(ValueError, match="No encryption key"):
            _ = config.key

    def test_key_property_loads_from_key_file(self):
        """key property loads from key_file if _key is empty."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".key", delete=False) as f:
            f.write("key-loaded-from-file")
            f.flush()
            key_path = f.name

        try:
            config = EncryptionConfig(enabled=True, key="", key_file=key_path)
            assert config.key == "key-loaded-from-file"
        finally:
            os.unlink(key_path)


class TestGenerateKey:
    """Tests for key generation."""

    def test_generate_key_returns_valid_key(self):
        """generate_key returns a string that works with FernetEncryptor."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        key = generate_key()
        # Should not raise
        encryptor = FernetEncryptor(key)
        encrypted = encryptor.encrypt("test")
        assert encryptor.decrypt_str(encrypted) == "test"

    def test_generate_key_writes_to_file(self):
        """generate_key writes key to specified file."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = os.path.join(tmpdir, "mykey")
            key = generate_key(key_path)

            assert os.path.exists(key_path)
            loaded = load_key_from_file(key_path)
            assert loaded == key


class TestEncryptionIntegrationSQLite:
    """Integration tests for encrypted SQLite storage."""

    def test_sqlite_store_and_retrieve_with_fernet_encryption(self):
        """Store and retrieve memories with Fernet field-level encryption."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "encrypted.db")
            key = generate_key()
            config = EncryptionConfig(enabled=True, mode="fernet", key=key)
            adapter = SQLiteStorageAdapter(db_path=db_path, encryption=config)

            memory = MemoryObject(
                memory_id="enc-test-1",
                user_id="user_enc",
                content="This content is encrypted at rest",
                embedding=[0.1, 0.2, 0.3],
                score=0.0,
                source=MemorySource.USER_STATED,
                importance=0.7,
                lifecycle_state=LifecycleState.ACTIVE,
                metadata={"secret_tag": "classified", "level": 5},
                embedding_dim=3,
                tags=["encrypted", "test"],
                confidence=1.0,
                memory_type=MemoryType.EPISODIC,
                namespace="default",
                version=1,
            )

            adapter.store(memory)
            retrieved = adapter.get("enc-test-1")

            assert retrieved is not None
            assert retrieved.content == "This content is encrypted at rest"
            assert retrieved.metadata == {"secret_tag": "classified", "level": 5}
            assert retrieved.user_id == "user_enc"
            assert retrieved.tags == ["encrypted", "test"]

    def test_sqlite_without_encryption_stores_plaintext(self):
        """SQLiteStorageAdapter without encryption stores plaintext (backward compat)."""
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "plain.db")
            adapter = SQLiteStorageAdapter(db_path=db_path)  # no encryption

            memory = MemoryObject(
                memory_id="plain-test-1",
                user_id="alice",
                content="Plaintext content",
                embedding=None,
                score=0.0,
                source=MemorySource.USER_STATED,
                importance=0.5,
                lifecycle_state=LifecycleState.ACTIVE,
                metadata={"plain": True},
                embedding_dim=None,
                tags=[],
                confidence=1.0,
                memory_type=MemoryType.EPISODIC,
                namespace="default",
                version=1,
            )

            adapter.store(memory)
            retrieved = adapter.get("plain-test-1")

            assert retrieved is not None
            assert retrieved.content == "Plaintext content"
            assert retrieved.user_id == "alice"

    def test_sqlite_store_many_with_encryption(self):
        """store_many works correctly with encryption enabled."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "many_enc.db")
            key = generate_key()
            config = EncryptionConfig(enabled=True, key=key)
            adapter = SQLiteStorageAdapter(db_path=db_path, encryption=config)

            memories = [
                MemoryObject(
                    memory_id=f"batch-{i}",
                    user_id="batch_user",
                    content=f"Batch memory number {i}",
                    embedding=[0.1] * 3,
                    score=0.0,
                    source=MemorySource.USER_STATED,
                    importance=0.5,
                    lifecycle_state=LifecycleState.ACTIVE,
                    metadata={"index": i},
                    embedding_dim=3,
                    tags=[],
                    confidence=1.0,
                    memory_type=MemoryType.EPISODIC,
                    namespace="default",
                    version=1,
                )
                for i in range(3)
            ]

            count = adapter.store_many(memories)
            assert count == 3

            for i in range(3):
                retrieved = adapter.get(f"batch-{i}")
                assert retrieved is not None
                assert retrieved.content == f"Batch memory number {i}"
                assert retrieved.metadata["index"] == i

    def test_sqlite_search_with_encryption(self):
        """search returns correctly decrypted results with encryption enabled."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "search_enc.db")
            key = generate_key()
            config = EncryptionConfig(enabled=True, key=key)
            adapter = SQLiteStorageAdapter(db_path=db_path, encryption=config)

            # Store a memory
            memory = MemoryObject(
                memory_id="search-enc-1",
                user_id="search_user",
                content="Python programming tutorial",
                embedding=[0.5, 0.3, 0.8],
                score=0.0,
                source=MemorySource.USER_STATED,
                importance=0.6,
                lifecycle_state=LifecycleState.ACTIVE,
                metadata={"language": "python"},
                embedding_dim=3,
                tags=["coding", "tutorial"],
                confidence=1.0,
                memory_type=MemoryType.EPISODIC,
                namespace="default",
                version=1,
            )
            adapter.store(memory)

            # Search (uses embedding similarity)
            results = adapter.search(
                user_id="search_user",
                query_embedding=[0.5, 0.3, 0.8],
                top_k=5,
            )

            assert len(results) >= 1
            found = next((r for r in results if r.memory_id == "search-enc-1"), None)
            assert found is not None
            assert found.content == "Python programming tutorial"


class TestEncryptionIntegrationJSON:
    """Integration tests for encrypted JSON storage."""

    def test_json_store_and_retrieve_with_encryption(self):
        """Store and retrieve memories from JSON adapter with encryption."""
        if not is_cryptography_available():
            pytest.skip("cryptography not installed")

        from kemi.adapters.storage.json import JSONStorageAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "encrypted.json")
            key = generate_key()
            config = EncryptionConfig(enabled=True, key=key)
            adapter = JSONStorageAdapter(path=json_path, encryption=config)

            memory = MemoryObject(
                memory_id="json-enc-1",
                user_id="json_enc_user",
                content="JSON encrypted content",
                embedding=None,
                score=0.0,
                source=MemorySource.USER_STATED,
                importance=0.8,
                lifecycle_state=LifecycleState.ACTIVE,
                metadata={"json_encrypted": True},
                embedding_dim=None,
                tags=["json", "test"],
                confidence=1.0,
                memory_type=MemoryType.SEMANTIC,
                namespace="default",
                version=1,
            )

            adapter.store(memory)
            retrieved = adapter.get("json-enc-1")

            assert retrieved is not None
            assert retrieved.content == "JSON encrypted content"
            assert retrieved.metadata == {"json_encrypted": True}
            assert retrieved.user_id == "json_enc_user"

    def test_json_without_encryption_backward_compat(self):
        """JSONStorageAdapter without encryption works as before."""
        from kemi.adapters.storage.json import JSONStorageAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "plain.json")
            adapter = JSONStorageAdapter(path=json_path)  # no encryption

            memory = MemoryObject(
                memory_id="json-plain-1",
                user_id="alice",
                content="Plain JSON content",
                embedding=None,
                score=0.0,
                source=MemorySource.USER_STATED,
                importance=0.5,
                lifecycle_state=LifecycleState.ACTIVE,
                metadata={},
                embedding_dim=None,
                tags=[],
                confidence=1.0,
                memory_type=MemoryType.EPISODIC,
                namespace="default",
                version=1,
            )

            adapter.store(memory)
            retrieved = adapter.get("json-plain-1")

            assert retrieved is not None
            assert retrieved.content == "Plain JSON content"


class TestSQLCipherManager:
    """Tests for SQLCipher full-database encryption."""

    def test_sqlcipher_manager_requires_key(self):
        """SQLCipherManager raises ValueError without key or key_file."""
        from kemi.encryption import SQLCipherManager

        with pytest.raises(ValueError, match="requires a key"):
            SQLCipherManager()

    def test_sqlcipher_manager_with_key(self):
        """SQLCipherManager accepts key parameter."""
        from kemi.encryption import SQLCipherManager

        manager = SQLCipherManager(key="test-key-12345")
        assert manager.key == "test-key-12345"

    def test_sqlcipher_manager_key_file(self):
        """SQLCipherManager loads key from file."""
        from kemi.encryption import SQLCipherManager

        with tempfile.NamedTemporaryFile(mode="w", suffix=".key", delete=False) as f:
            f.write("sqlcipher-test-key")
            f.flush()
            key_path = f.name

        try:
            manager = SQLCipherManager(key_file=key_path)
            assert manager.key == "sqlcipher-test-key"
        finally:
            os.unlink(key_path)


class TestEncryptionAvailability:
    """Tests for encryption package availability checks."""

    def test_is_cryptography_available_returns_bool(self):
        """is_cryptography_available returns a boolean."""
        result = is_cryptography_available()
        assert isinstance(result, bool)

    def test_is_sqlcipher_available_returns_bool(self):
        """is_sqlcipher_available returns a boolean."""
        result = is_sqlcipher_available()
        assert isinstance(result, bool)