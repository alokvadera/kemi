"""Tests for src/kemi/api_keys.py"""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from kemi.exceptions import ConfigurationError
from kemi.infra.api_keys import (
    KEY_PREFIX,
    APIKey,
    APIKeyManager,
    _generate_key_id,
    _generate_raw_key,
    _hash_key,
    make_expiry,
)


@pytest.fixture
def conn(tmp_path):
    import sqlite3

    db_path = str(tmp_path / "api_keys.db")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE api_keys (
            key_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            hashed_key TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            last_used_at TEXT,
            revoked_at TEXT
        )
    """)
    connection.commit()
    return connection


@pytest.fixture
def mgr(conn):
    return APIKeyManager(connection=conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_generate_raw_key_format(self):
        key = _generate_raw_key()
        assert key.startswith(KEY_PREFIX)
        assert len(key) > len(KEY_PREFIX)

    def test_generate_key_id_format(self):
        kid = _generate_key_id()
        assert kid.startswith("kmi_")
        assert len(kid) == 12  # "kmi_" + 8 hex chars

    def test_hash_key_format(self):
        raw = "kemi_test_key_123"
        hashed = _hash_key(raw)
        parts = hashed.split(":")
        assert len(parts) == 2  # fast_hash : bcrypt_hash
        assert all(c in "0123456789abcdef" for c in parts[0])
        assert len(parts[0]) == 64  # HMAC-SHA256 hex
        assert parts[1].startswith("$2")  # bcrypt hash prefix ($2b$, $2a$, $2y$)

    def test_hash_key_verify_roundtrip(self):
        from kemi.infra.api_keys import _verify_key
        raw = "kemi_test_key_123"
        hashed = _hash_key(raw)
        assert _verify_key(raw, hashed) is True
        assert _verify_key("wrong_key", hashed) is False

    def test_fast_hash_uses_pepper(self):
        from kemi.infra.api_keys import _fast_hash
        raw = "kemi_test_key_123"
        h1 = _fast_hash(raw)
        assert len(h1) == 64
        # Same raw key with same pepper must produce same fast_hash
        h2 = _fast_hash(raw)
        assert h1 == h2
        # Different raw key must produce different fast_hash
        h3 = _fast_hash("kemi_different_key")
        assert h1 != h3

    def test_make_expiry_none(self):
        assert make_expiry(None) is None

    def test_make_expiry_days(self):
        result = make_expiry(7)
        assert isinstance(result, datetime)
        now = datetime.now(timezone.utc)
        assert now < result <= now + timedelta(days=8)


# ---------------------------------------------------------------------------
# APIKey dataclass
# ---------------------------------------------------------------------------

class TestAPIKey:
    def test_is_expired_no_expiry(self):
        key = APIKey(
            key_id="kmi_123",
            user_id="alice",
            name="test",
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=None,
            last_used_at=None,
            revoked_at=None,
        )
        assert not key.is_expired()

    def test_is_expired_with_future_expiry(self):
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        key = APIKey(
            key_id="kmi_123",
            user_id="alice",
            name="test",
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=future,
            last_used_at=None,
            revoked_at=None,
        )
        assert not key.is_expired()

    def test_is_expired_with_past_expiry(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        key = APIKey(
            key_id="kmi_123",
            user_id="alice",
            name="test",
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=past,
            last_used_at=None,
            revoked_at=None,
        )
        assert key.is_expired()

    def test_is_expired_naive_datetime(self):
        """Naive expiry should be treated as UTC."""
        past = (datetime.utcnow() - timedelta(days=1)).isoformat()
        key = APIKey(
            key_id="kmi_123",
            user_id="alice",
            name="test",
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=past,
            last_used_at=None,
            revoked_at=None,
        )
        assert key.is_expired()

    def test_is_expired_invalid_date(self):
        key = APIKey(
            key_id="kmi_123",
            user_id="alice",
            name="test",
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at="not-a-date",
            last_used_at=None,
            revoked_at=None,
        )
        assert not key.is_expired()

    def test_is_active_true(self):
        key = APIKey(
            key_id="kmi_123",
            user_id="alice",
            name="test",
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=None,
            last_used_at=None,
            revoked_at=None,
        )
        assert key.is_active()

    def test_is_active_revoked(self):
        key = APIKey(
            key_id="kmi_123",
            user_id="alice",
            name="test",
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=None,
            last_used_at=None,
            revoked_at=datetime.now(timezone.utc).isoformat(),
        )
        assert not key.is_active()

    def test_is_active_expired(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        key = APIKey(
            key_id="kmi_123",
            user_id="alice",
            name="test",
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=past,
            last_used_at=None,
            revoked_at=None,
        )
        assert not key.is_active()

    def test_to_dict_excludes_secret_by_default(self):
        key = APIKey(
            key_id="kmi_123",
            user_id="alice",
            name="test",
            created_at="2026-01-01T00:00:00+00:00",
            expires_at=None,
            last_used_at=None,
            revoked_at=None,
            raw_key="kemi_secret",
        )
        d = key.to_dict()
        assert "api_key" not in d
        assert d["key_id"] == "kmi_123"
        assert d["user_id"] == "alice"

    def test_to_dict_includes_secret_when_requested(self):
        key = APIKey(
            key_id="kmi_123",
            user_id="alice",
            name="test",
            created_at="2026-01-01T00:00:00+00:00",
            expires_at=None,
            last_used_at=None,
            revoked_at=None,
            raw_key="kemi_secret",
        )
        d = key.to_dict(include_secret=True)
        assert d["api_key"] == "kemi_secret"


# ---------------------------------------------------------------------------
# APIKeyManager.create_key
# ---------------------------------------------------------------------------

class TestCreateKey:
    def test_create_key_basic(self, mgr):
        key = mgr.create_key(user_id="alice", name="laptop")
        assert key.user_id == "alice"
        assert key.name == "laptop"
        assert key.raw_key is not None
        assert key.raw_key.startswith(KEY_PREFIX)
        assert key.created_at is not None
        assert key.expires_at is None
        assert key.revoked_at is None

    def test_create_key_with_expiry(self, mgr):
        expires = datetime.now(timezone.utc) + timedelta(days=7)
        key = mgr.create_key(user_id="alice", name="temp", expires_at=expires)
        assert key.expires_at is not None

    def test_create_key_empty_user_id_raises(self, mgr):
        with pytest.raises(ValueError, match="user_id and name are required"):
            mgr.create_key(user_id="", name="test")

    def test_create_key_empty_name_raises(self, mgr):
        with pytest.raises(ValueError, match="user_id and name are required"):
            mgr.create_key(user_id="alice", name="")

    def test_create_key_duplicate_hash_raises(self, mgr, monkeypatch):
        """Simulate an IntegrityError by creating a unique index violation."""
        # First create a key normally
        mgr.create_key(user_id="alice", name="first")

        # Patch _generate_raw_key to return a fixed value and _hash_key
        # to return a fixed hash so we can force a collision.
        monkeypatch.setattr("kemi.infra.api_keys._generate_raw_key", lambda: "kemi_collision_test")
        monkeypatch.setattr(
            "kemi.infra.api_keys._hash_key",
            lambda _raw: "fixed_hash_for_collision",
        )

        # Pre-insert a row with the same hashed_key to force collision
        mgr._conn.execute(
            "INSERT INTO api_keys (key_id, user_id, hashed_key, name, created_at) VALUES (?, ?, ?, ?, ?)",  # noqa: E501
            ("kmi_collide", "bob", "fixed_hash_for_collision", "collision", datetime.now(timezone.utc).isoformat()),  # noqa: E501
        )
        mgr._conn.commit()

        with pytest.raises(ConfigurationError, match="Failed to create API key"):
            mgr.create_key(user_id="alice", name="second")


# ---------------------------------------------------------------------------
# APIKeyManager.lookup
# ---------------------------------------------------------------------------

class TestLookup:
    def test_lookup_found(self, mgr):
        key = mgr.create_key(user_id="alice", name="laptop")
        found = mgr.lookup(key.raw_key)
        assert found is not None
        assert found.key_id == key.key_id
        assert found.user_id == "alice"

    def test_lookup_not_found(self, mgr):
        assert mgr.lookup("kemi_nonexistent_key_xyz123") is None

    def test_lookup_invalid_prefix(self, mgr):
        assert mgr.lookup("invalid_prefix") is None

    def test_lookup_empty_string(self, mgr):
        assert mgr.lookup("") is None

    def test_lookup_revoked_key_returns_none(self, mgr):
        key = mgr.create_key(user_id="alice", name="revoked")
        mgr.revoke(key.key_id)
        assert mgr.lookup(key.raw_key) is None

    def test_lookup_expired_key_returns_none(self, mgr):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        key = mgr.create_key(user_id="alice", name="expired", expires_at=past)
        assert mgr.lookup(key.raw_key) is None

    def test_lookup_does_not_update_last_used_at(self, mgr):
        """Per-request last_used_at write was removed to avoid SQLite serialization."""
        key = mgr.create_key(user_id="alice", name="track")
        assert key.last_used_at is None

        mgr.lookup(key.raw_key)

        # Reload from DB to verify last_used_at was NOT updated
        row = mgr._conn.execute(
            "SELECT last_used_at FROM api_keys WHERE key_id = ?",
            (key.key_id,),
        ).fetchone()
        assert row["last_used_at"] is None

    def test_lookup_active_true(self, mgr):
        key = mgr.create_key(user_id="alice", name="active")
        found = mgr.lookup(key.raw_key)
        assert found is not None
        assert found.is_active()

    def test_lookup_legacy_hash_still_works(self, mgr):
        """Backward-compat: keys stored with old hash formats
        must still be findable and verifiable."""
        # Very legacy: plain unsalted SHA-256 (64 hex chars)
        raw_key_v1 = "kemi_legacy_v1_key_12345"
        legacy_hash_v1 = hashlib.sha256(raw_key_v1.encode("utf-8")).hexdigest()
        mgr._conn.execute(
            "INSERT INTO api_keys (key_id, user_id, hashed_key, name, created_at) VALUES (?, ?, ?, ?, ?)",  # noqa: E501
            ("kmi_legacy_v1", "alice", legacy_hash_v1, "legacy_v1", datetime.now(timezone.utc).isoformat()),
        )
        mgr._conn.commit()
        found = mgr.lookup(raw_key_v1)
        assert found is not None
        assert found.key_id == "kmi_legacy_v1"
        assert found.is_active()

        # Legacy PBKDF2: fast_hash:salt_hex:pbkdf2_hash
        raw_key_v2 = "kemi_legacy_v2_key_12345"
        salt = b"\x00" * 32
        pbkdf2_hash = hashlib.pbkdf2_hmac("sha256", raw_key_v2.encode("utf-8"), salt, 600000)
        fast_hash_v2 = hashlib.sha256(raw_key_v2.encode("utf-8")).hexdigest()
        legacy_hash_v2 = f"{fast_hash_v2}:{salt.hex()}:{pbkdf2_hash.hex()}"
        mgr._conn.execute(
            "INSERT INTO api_keys (key_id, user_id, hashed_key, name, created_at) VALUES (?, ?, ?, ?, ?)",  # noqa: E501
            ("kmi_legacy_v2", "alice", legacy_hash_v2, "legacy_v2", datetime.now(timezone.utc).isoformat()),
        )
        mgr._conn.commit()
        found = mgr.lookup(raw_key_v2)
        assert found is not None
        assert found.key_id == "kmi_legacy_v2"
        assert found.is_active()


# ---------------------------------------------------------------------------
# APIKeyManager.list_keys
# ---------------------------------------------------------------------------

class TestListKeys:
    def test_list_all_keys(self, mgr):
        mgr.create_key(user_id="alice", name="key1")
        mgr.create_key(user_id="bob", name="key2")
        keys = mgr.list_keys()
        assert len(keys) == 2

    def test_list_keys_filtered_by_user(self, mgr):
        mgr.create_key(user_id="alice", name="key1")
        mgr.create_key(user_id="alice", name="key2")
        mgr.create_key(user_id="bob", name="key3")
        keys = mgr.list_keys(user_id="alice")
        assert len(keys) == 2
        assert all(k.user_id == "alice" for k in keys)

    def test_list_keys_excludes_raw_key(self, mgr):
        mgr.create_key(user_id="alice", name="key1")
        keys = mgr.list_keys()
        assert all(k.raw_key is None for k in keys)

    def test_list_keys_sorted_desc(self, mgr):
        import time

        mgr.create_key(user_id="alice", name="older")
        time.sleep(0.01)
        mgr.create_key(user_id="alice", name="newer")
        keys = mgr.list_keys()
        assert keys[0].name == "newer"
        assert keys[1].name == "older"

    def test_list_keys_empty(self, mgr):
        assert mgr.list_keys() == []


# ---------------------------------------------------------------------------
# APIKeyManager.revoke
# ---------------------------------------------------------------------------

class TestRevoke:
    def test_revoke_existing(self, mgr):
        key = mgr.create_key(user_id="alice", name="revoke_me")
        assert mgr.revoke(key.key_id) is True

        # Verify in DB
        row = mgr._conn.execute(
            "SELECT revoked_at FROM api_keys WHERE key_id = ?",
            (key.key_id,),
        ).fetchone()
        assert row["revoked_at"] is not None

    def test_revoke_nonexistent(self, mgr):
        assert mgr.revoke("kmi_nonexistent") is False

    def test_revoke_already_revoked(self, mgr):
        key = mgr.create_key(user_id="alice", name="already")
        mgr.revoke(key.key_id)
        assert mgr.revoke(key.key_id) is False


# ---------------------------------------------------------------------------
# APIKeyManager.get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_existing(self, mgr):
        key = mgr.create_key(user_id="alice", name="get_me")
        found = mgr.get(key.key_id)
        assert found is not None
        assert found.key_id == key.key_id
        assert found.raw_key is None

    def test_get_nonexistent(self, mgr):
        assert mgr.get("kmi_nonexistent") is None


# ---------------------------------------------------------------------------
# APIKeyManager.cleanup_expired
# ---------------------------------------------------------------------------

class TestCleanupExpired:
    def test_cleanup_expired_keys(self, mgr):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        future = datetime.now(timezone.utc) + timedelta(days=1)

        expired = mgr.create_key(user_id="alice", name="expired", expires_at=past)
        active = mgr.create_key(user_id="alice", name="active", expires_at=future)
        no_expiry = mgr.create_key(user_id="alice", name="no_expiry")

        count = mgr.cleanup_expired()
        assert count == 1

        # Verify expired key is revoked
        row = mgr._conn.execute(
            "SELECT revoked_at FROM api_keys WHERE key_id = ?",
            (expired.key_id,),
        ).fetchone()
        assert row["revoked_at"] is not None

        # Active and no-expiry keys should not be revoked
        assert mgr.get(active.key_id).revoked_at is None
        assert mgr.get(no_expiry.key_id).revoked_at is None

    def test_cleanup_no_expired_keys(self, mgr):
        mgr.create_key(user_id="alice", name="active")
        assert mgr.cleanup_expired() == 0

    def test_cleanup_already_revoked_not_counted(self, mgr):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        key = mgr.create_key(user_id="alice", name="expired", expires_at=past)
        mgr.revoke(key.key_id)
        # Already revoked, so cleanup should not count it
        assert mgr.cleanup_expired() == 0
