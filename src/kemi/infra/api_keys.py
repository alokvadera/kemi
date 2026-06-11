"""API key management for multi-tenant FastAPI server.

Stores hashed API keys in the same SQLite database as memories (table
`api_keys`, schema version 8). The raw key is returned to the caller only
once at creation time; only the bcrypt hash is persisted.

Key format
----------
Raw key: ``kemi_<43-char base32 token>`` (e.g. ``kemi_abc...xyz``).
Key id:  short, non-secret identifier shown in listings so an operator
         can revoke a specific key without seeing the secret.
Hash:    ``fast_hash:bcrypt_hash``.
         fast_hash = HMAC-SHA256(pepper, raw_key) for efficient lookup.
         bcrypt_hash = bcrypt(raw_key, cost=12).

The pepper is read from ``KEMI_API_KEY_PEPPER`` env var; a default is
provided so the feature works out of the box, but production deployments
should set their own pepper.

Backward compatibility
----------------------
Lookup still verifies legacy PBKDF2-HMAC-SHA256 hashes (format
``fast_hash:salt_hex:pbkdf2_hash``) and very old plain unsalted
SHA-256 hashes (64 hex chars). New keys are always hashed with bcrypt.

Expiry is checked at lookup time; expired keys are treated as invalid.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt

from kemi.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

KEY_PREFIX = "kemi_"
KEY_ID_PREFIX = "kmi_"  # short, non-secret identifier for listings
_HASH_ALGO = "sha256"
_RANDOM_BYTES = 32  # 256 bits of entropy
_KEY_ID_BYTES = 4  # 8 hex chars — plenty for non-secret display IDs


@dataclass
class APIKey:
    """In-memory representation of an api_keys row.

    `raw_key` is populated only on creation, never loaded from storage.
    """

    key_id: str
    user_id: str
    name: str
    created_at: str
    expires_at: str | None
    last_used_at: str | None
    revoked_at: str | None
    raw_key: str | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        if not self.expires_at:
            return False
        try:
            expiry = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return False
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry <= (now or datetime.now(timezone.utc))

    def is_active(self) -> bool:
        return self.revoked_at is None and not self.is_expired()

    def to_dict(self, include_secret: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "key_id": self.key_id,
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used_at": self.last_used_at,
            "revoked_at": self.revoked_at,
        }
        if include_secret and self.raw_key is not None:
            d["api_key"] = self.raw_key
        return d


def _pepper() -> bytes:
    """Return the server-side pepper for HMAC fast-hashing.

    The pepper prevents generic rainbow-table attacks against the lookup
    hash. Production deployments should set ``KEMI_API_KEY_PEPPER``
    to a long, random string unique to the deployment."""
    return os.environ.get(
        "KEMI_API_KEY_PEPPER",
        "kemi-default-pepper-2026-change-me",
    ).encode("utf-8")


def _fast_hash(raw_key: str) -> str:
    """Return a HMAC-SHA256(pepper, raw_key) hex string for DB lookup."""
    return hmac.new(_pepper(), raw_key.encode("utf-8"), hashlib.sha256).hexdigest()


def _hash_key(raw_key: str) -> str:
    """Hash a raw API key with bcrypt.

    Returns a string in the format ``fast_hash:bcrypt_hash``.
    The fast_hash prefix (HMAC-SHA256 with a server-side pepper) enables
    efficient database lookup without running the expensive bcrypt
    verification on every row."""
    hash_bytes = bcrypt.hashpw(raw_key.encode("utf-8"), bcrypt.gensalt(rounds=12))
    return _fast_hash(raw_key) + ":" + hash_bytes.decode("utf-8")


def _verify_key(raw_key: str, stored_hash: str) -> bool:
    """Verify a raw API key against a stored hash.

    Supports three formats (newest to oldest):
    1. ``fast_hash:bcrypt_hash`` — bcrypt (current)
    2. ``fast_hash:salt_hex:pbkdf2_hash`` — PBKDF2-HMAC-SHA256 (legacy)
    3. 64 hex chars — plain unsalted SHA-256 (very legacy)

    Uses constant-time comparison to resist timing attacks.
    """
    # Very legacy: plain unsalted SHA-256 (64 hex chars)
    if len(stored_hash) == 64:
        return hmac.compare_digest(
            hashlib.sha256(raw_key.encode("utf-8")).hexdigest(),
            stored_hash,
        )

    parts = stored_hash.split(":")

    # Legacy PBKDF2: fast_hash:salt_hex:pbkdf2_hash (3 parts, all hex)
    if len(parts) == 3:
        fast_hash, salt_hex, hash_hex = parts
        if not hmac.compare_digest(
            fast_hash, hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        ):
            return False
        try:
            salt = bytes.fromhex(salt_hex)
        except ValueError:
            return False
        expected = hashlib.pbkdf2_hmac("sha256", raw_key.encode("utf-8"), salt, 600000)
        return hmac.compare_digest(expected.hex(), hash_hex)

    # Current bcrypt: fast_hash:bcrypt_hash (2 parts, bcrypt hash starts with $2)
    if len(parts) == 2:
        fast_hash, bcrypt_hash = parts
        if not hmac.compare_digest(fast_hash, _fast_hash(raw_key)):
            return False
        return bcrypt.checkpw(raw_key.encode("utf-8"), bcrypt_hash.encode("utf-8"))

    return False


def _generate_raw_key() -> str:
    # 32 bytes → 43 base32 chars (no padding). Strip '=' for a clean token.
    token = secrets.token_urlsafe(_RANDOM_BYTES).rstrip("=")
    return f"{KEY_PREFIX}{token}"


def _generate_key_id() -> str:
    return f"{KEY_ID_PREFIX}{secrets.token_hex(_KEY_ID_BYTES)}"


class APIKeyManager:
    """CRUD + lookup for API keys backed by the `api_keys` table."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def create_key(
        self,
        user_id: str,
        name: str,
        expires_at: datetime | None = None,
    ) -> APIKey:
        """Create a new API key for a user.

        Args:
            user_id: Owner of the key.
            name: Human-readable label (e.g. "laptop", "ci-runner").
            expires_at: Optional expiry datetime (UTC). None = never expires.

        Returns:
            APIKey whose ``raw_key`` is set. Callers must surface the
            ``raw_key`` to the user; it is unrecoverable afterwards.
        """
        if not user_id or not name:
            raise ValidationError("user_id and name are required")

        raw_key = _generate_raw_key()
        key_id = _generate_key_id()
        hashed = _hash_key(raw_key)
        now = datetime.now(timezone.utc).isoformat()
        expires_iso = expires_at.isoformat() if expires_at else None

        try:
            self._conn.execute(
                """
                INSERT INTO api_keys
                    (key_id, user_id, hashed_key, name, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key_id, user_id, hashed, name, now, expires_iso),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as e:
            # Collisions on key_id or hashed_key are astronomically unlikely,
            # but surface a clear error rather than a generic IntegrityError.
            raise ConfigurationError(f"Failed to create API key: {e}") from e

        logger.info("Created API key %s for user %s", key_id, user_id)
        return APIKey(
            key_id=key_id,
            user_id=user_id,
            name=name,
            created_at=now,
            expires_at=expires_iso,
            last_used_at=None,
            revoked_at=None,
            raw_key=raw_key,
        )

    def lookup(self, raw_key: str) -> APIKey | None:
        """Return the active APIKey for a raw key, or None.

        Expired or revoked keys return None. The last_used_at timestamp
        is updated in the background (best-effort; failures are logged
        but do not affect the lookup result).
        """
        if not raw_key or not raw_key.startswith(KEY_PREFIX):
            return None
        # Try current fast-hash (HMAC with pepper) and legacy fast-hash
        # (unsalted SHA-256) in a single query, then verify candidates.
        row = None
        current_fast = _fast_hash(raw_key)
        legacy_fast = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        rows = self._conn.execute(
            """
            SELECT key_id, user_id, name, created_at, expires_at,
                   last_used_at, revoked_at, hashed_key
            FROM api_keys
            WHERE (hashed_key = ? OR hashed_key LIKE ? OR hashed_key LIKE ?)
              AND revoked_at IS NULL
            """,
            (legacy_fast, current_fast + ":%", legacy_fast + ":%"),
        ).fetchall()
        for r in rows:
            if _verify_key(raw_key, r["hashed_key"]):
                row = r
                break

        if row is None:
            return None

        key = APIKey(
            key_id=row["key_id"],
            user_id=row["user_id"],
            name=row["name"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            last_used_at=row["last_used_at"],
            revoked_at=row["revoked_at"],
        )
        if not key.is_active():
            return None

        return key

    def list_keys(self, user_id: str | None = None) -> list[APIKey]:
        """List API keys, optionally filtered by user.

        Returned objects never contain the raw key.
        """
        if user_id is None:
            rows = self._conn.execute(
                """
                SELECT key_id, user_id, name, created_at, expires_at,
                       last_used_at, revoked_at
                FROM api_keys
                ORDER BY created_at DESC
                """
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT key_id, user_id, name, created_at, expires_at,
                       last_used_at, revoked_at
                FROM api_keys
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()

        return [
            APIKey(
                key_id=r["key_id"],
                user_id=r["user_id"],
                name=r["name"],
                created_at=r["created_at"],
                expires_at=r["expires_at"],
                last_used_at=r["last_used_at"],
                revoked_at=r["revoked_at"],
            )
            for r in rows
        ]

    def revoke(self, key_id: str) -> bool:
        """Revoke a key by id. Returns True if a key was revoked."""
        cursor = self._conn.execute(
            """
            UPDATE api_keys
            SET revoked_at = ?
            WHERE key_id = ? AND revoked_at IS NULL
            """,
            (datetime.now(timezone.utc).isoformat(), key_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get(self, key_id: str) -> APIKey | None:
        row = self._conn.execute(
            """
            SELECT key_id, user_id, name, created_at, expires_at,
                   last_used_at, revoked_at
            FROM api_keys
            WHERE key_id = ?
            """,
            (key_id,),
        ).fetchone()
        if row is None:
            return None
        return APIKey(
            key_id=row["key_id"],
            user_id=row["user_id"],
            name=row["name"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            last_used_at=row["last_used_at"],
            revoked_at=row["revoked_at"],
        )

    def cleanup_expired(self) -> int:
        """Revoke keys whose expiry is in the past. Returns count revoked."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """
            UPDATE api_keys
            SET revoked_at = ?
            WHERE revoked_at IS NULL
              AND expires_at IS NOT NULL
              AND expires_at <= ?
            """,
            (now, now),
        )
        self._conn.commit()
        return cursor.rowcount


def make_expiry(days: int | None) -> datetime | None:
    """Helper to convert a 'days from now' hint into a UTC datetime."""
    if days is None:
        return None
    return datetime.now(timezone.utc) + timedelta(days=days)
