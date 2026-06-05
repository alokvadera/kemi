"""Tests for src/kemi/api_keys.py"""

from datetime import datetime, timedelta, timezone

import pytest

from kemi.api_keys import (
    APIKey,
    APIKeyManager,
    KEY_PREFIX,
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

    def test_hash_key_is_sha256_hex(self):
        raw = "kemi_test_key_123"
        hashed = _hash_key(raw)
        assert len(hashed) == 64
        assert all(c in "0123456789abcdef" for c in hashed)

    def test_hash_key_deterministic(self):
        raw = "kemi_test_key_123"
        assert _hash_key(raw) == _hash_key(raw)

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

        # Patch _generate_raw_key to return the same value again
        # This is extremely unlikely naturally, but we can force it
        # by directly inserting a duplicate hashed_key instead.
        from kemi.api_keys import _generate_raw_key as orig_gen
        monkeypatch.setattr("kemi.api_keys._generate_raw_key", lambda: "kemi_collision_test")

        # Pre-insert a row with the same hashed_key to force collision
        hashed = _hash_key("kemi_collision_test")
        mgr._conn.execute(
            "INSERT INTO api_keys (key_id, user_id, hashed_key, name, created_at) VALUES (?, ?, ?, ?, ?)",
            ("kmi_collide", "bob", hashed, "collision", datetime.now(timezone.utc).isoformat()),
        )
        mgr._conn.commit()

        with pytest.raises(RuntimeError, match="Failed to create API key"):
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

    def test_lookup_updates_last_used_at(self, mgr):
        key = mgr.create_key(user_id="alice", name="track")
        assert key.last_used_at is None

        mgr.lookup(key.raw_key)

        # Reload from DB to verify last_used_at was updated
        row = mgr._conn.execute(
            "SELECT last_used_at FROM api_keys WHERE key_id = ?",
            (key.key_id,),
        ).fetchone()
        assert row["last_used_at"] is not None

    def test_lookup_active_true(self, mgr):
        key = mgr.create_key(user_id="alice", name="active")
        found = mgr.lookup(key.raw_key)
        assert found is not None
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
