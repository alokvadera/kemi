"""
Encryption layer for kemi storage adapters.

Supports two encryption approaches:
- Approach A: SQLCipher full-database encryption for SQLite
- Approach B: Fernet field-level encryption for all adapters (content + metadata)

Key management:
- SQLCipher: key loaded from --key-file path passed to init
- Fernet: key loaded from KEMI_ENCRYPTION_KEY env var or --key-file path
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from kemi.exceptions import ConfigurationError, EncryptionError

__all__ = [
    "FernetEncryptor",
    "SQLCipherManager",
    "EncryptionConfig",
    "FieldEncryptor",
]


class EncryptionConfig:
    """Configuration for encryption. Passed to storage adapters."""

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "fernet",  # "fernet" or "sqlcipher"
        key: str | None = None,
        key_file: str | None = None,
        key_id: str | None = None,
        encrypt_user_id: bool = False,
        encrypt_session_id: bool = False,
    ) -> None:
        self.enabled = enabled
        self.mode = mode  # "fernet" or "sqlcipher"
        self._key = key or ""
        self.key_file = key_file
        self.key_id = key_id or "default"
        self.encrypt_user_id = encrypt_user_id
        self.encrypt_session_id = encrypt_session_id

    @classmethod
    def from_env(cls) -> EncryptionConfig:
        """Load encryption config from environment variables."""
        enabled = os.environ.get("KEMI_ENCRYPTION_ENABLED", "").lower() in ("1", "true", "yes")
        mode = os.environ.get("KEMI_ENCRYPTION_MODE", "fernet")
        key = os.environ.get("KEMI_ENCRYPTION_KEY", "")
        key_id = os.environ.get("KEMI_ENCRYPTION_KEY_ID", "default")
        return cls(enabled=enabled, mode=mode, key=key, key_id=key_id)

    @classmethod
    def from_key_file(cls, path: str, key_id: str | None = None) -> EncryptionConfig:
        """Load encryption config from a key file."""
        key = load_key_from_file(path)
        kid = key_id if key_id is not None else "default"
        return cls(enabled=True, mode="fernet", key=key, key_file=path, key_id=kid)

    @property
    def key(self) -> str:
        if self._key:
            return self._key
        if self.key_file:
            return load_key_from_file(self.key_file)
        raise EncryptionError("No encryption key configured. Set KEMI_ENCRYPTION_KEY env var or pass --key-file")  # noqa: E501


def load_key_from_file(path: str) -> str:
    """Load encryption key from a file."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Key file not found: {path}")
    return p.read_text().strip()


def generate_key(path: str | None = None) -> str:
    """Generate a new Fernet-compatible encryption key.

    Uses Fernet.generate_key() which produces a 128-bit URL-safe base64-encoded
    key (43 bytes). If path is provided, write the key to that file.
    Returns the key as a string.
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError as e:
        raise ConfigurationError(
            "cryptography package required for key generation. "
            "Install with: pip install kemi[encryption] or pip install cryptography"
        ) from e

    key = Fernet.generate_key().decode("utf-8")
    if path:
        p = Path(path).expanduser()
        p.write_text(key + "\n")
    return key


# ---------------------------------------------------------------------------
# Fernet field-level encryption
# ---------------------------------------------------------------------------

class FernetEncryptor:
    """Fernet symmetric encryption for field-level data protection.

    Fernet is a standard symmetric encryption method (AES-128-CBC with HMAC).
    Encrypts arbitrary bytes and encodes them as URL-safe base64.
    """

    def __init__(self, key: str, salt: bytes | None = None) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as e:
            raise ConfigurationError(
                "cryptography package required for Fernet encryption. "
                "Install with: pip install kemi[encryption] or pip install cryptography"
            ) from e

        import hashlib

        if salt is not None:
            # Proper derivation: PBKDF2-HMAC-SHA256 with user-supplied salt.
            derived = hashlib.pbkdf2_hmac("sha256", key.encode("utf-8"), salt, 600000)
            fernet_key = base64.urlsafe_b64encode(derived)
            self._fernet = Fernet(fernet_key)
        else:
            # Try to use the key directly if it's already a valid Fernet key.
            try:
                self._fernet = Fernet(key.encode("utf-8"))
            except Exception:
                # Legacy fallback: single-round unsalted SHA-256.
                # This is weak and kept only for backward compatibility with
                # existing encrypted data.
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "FernetEncryptor: the provided key is not a valid Fernet key "
                    "(must be 32 bytes URL-safe base64). Falling back to a single-round "
                    "unsalted SHA-256 derivation. This is cryptographically weak — "
                    "generate a proper key with `kemi.encryption.generate_key()` and "
                    "set KEMI_ENCRYPTION_KEY to that value."
                )
                digest = hashlib.sha256(key.encode("utf-8")).digest()
                fernet_key = base64.urlsafe_b64encode(digest)
                self._fernet = Fernet(fernet_key)

    def encrypt(self, data: str | bytes) -> str:
        """Encrypt data, return base64-encoded ciphertext."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        result = self._fernet.encrypt(data)
        # cryptography Fernet.encrypt() returns bytes on some versions,
        # str on others — normalize to string
        if isinstance(result, bytes):
            result = result.decode("utf-8")
        return result

    def decrypt(self, ciphertext: str | bytes) -> bytes:
        """Decrypt base64-encoded ciphertext, return raw bytes."""
        if isinstance(ciphertext, str):
            ciphertext = ciphertext.encode("utf-8")
        return self._fernet.decrypt(ciphertext)

    def decrypt_str(self, ciphertext: str) -> str:
        """Decrypt ciphertext, return as string."""
        return self.decrypt(ciphertext).decode("utf-8")


class FieldEncryptor:
    """Encrypts and decrypts specific memory fields.

    Encrypted fields are stored as JSON-serialized objects:
        {"encrypted": true, "key_id": "...", "data": "...base64 ciphertext..."}

    Fields encrypted by default: content, metadata (JSON fields with sensitive data).
    Optionally encrypts: user_id, session_id.
    """

    ENCRYPTED_PREFIX = {"encrypted": True}

    def __init__(
        self,
        config: EncryptionConfig,
        encrypt_fields: list[str] | None = None,
        encrypt_user_id: bool | None = None,
        encrypt_session_id: bool | None = None,
    ) -> None:
        self._config = config
        if not config.enabled:
            self._fernet: FernetEncryptor | None = None
            self._encrypt_fields: frozenset[str] = frozenset()
            self._encrypt_user_id = False
            self._encrypt_session_id = False
            self._key_id = config.key_id
            return

        self._fernet = FernetEncryptor(config.key)
        self._encrypt_fields = frozenset(encrypt_fields or ["content", "metadata"])
        # Read from config when params are not explicitly set (default to None)
        self._encrypt_user_id = encrypt_user_id if encrypt_user_id is not None else getattr(config, "encrypt_user_id", False)  # noqa: E501
        self._encrypt_session_id = encrypt_session_id if encrypt_session_id is not None else getattr(config, "encrypt_session_id", False)  # noqa: E501
        self._key_id = config.key_id

    @property
    def is_enabled(self) -> bool:
        return self._fernet is not None

    def _encrypt_value(self, value: Any) -> dict[str, Any]:
        """Encrypt a value and return an encrypted envelope dict."""
        json_bytes = json.dumps(value).encode("utf-8")
        salt = os.urandom(16)
        encryptor = FernetEncryptor(self._config.key, salt=salt)
        encrypted = encryptor.encrypt(json_bytes)
        if isinstance(encrypted, bytes):
            encrypted = encrypted.decode("utf-8")
        return {
            **self.ENCRYPTED_PREFIX,
            "key_id": self._key_id,
            "salt": salt.hex(),
            "kdf": "pbkdf2",
            "data": encrypted,
        }

    def encrypt_field(self, field_name: str, value: Any) -> Any:
        """Encrypt a field value if encryption is enabled for it.

        Handles both standard encryptable fields (content, metadata, etc.)
        and optional extra fields like user_id and session_id.
        """
        if not self.is_enabled:
            return value
        if value is None:
            return None

        # Check standard encryptable fields
        if field_name in self._encrypt_fields:
            return self._encrypt_value(value)

        # Check optional extra fields (user_id, session_id)
        if field_name == "user_id" and self._encrypt_user_id:
            return self._encrypt_value(value)
        if field_name == "session_id" and self._encrypt_session_id:
            return self._encrypt_value(value)

        return value

    def decrypt_field(self, field_name: str, value: Any) -> Any:
        """Decrypt a field value if it's an encrypted blob."""
        if not self.is_enabled:
            return value
        if not self._is_encrypted(value):
            return value

        ciphertext = value.get("data", "")
        salt_hex = value.get("salt")
        kdf = value.get("kdf", "legacy")

        if salt_hex and kdf == "pbkdf2":
            decryptor = FernetEncryptor(
                self._config.key, salt=bytes.fromhex(salt_hex)
            )
            decrypted_bytes = decryptor.decrypt(ciphertext)
        else:
            decrypted_bytes = self._fernet.decrypt(ciphertext)
        return json.loads(decrypted_bytes.decode("utf-8"))

    def _is_encrypted(self, value: Any) -> bool:
        return isinstance(value, dict) and value.get("encrypted") is True

    def encrypt_memory_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Encrypt relevant fields in a memory row dict (before storage)."""
        if not self.is_enabled:
            return row

        result = dict(row)
        for field in self._encrypt_fields:
            if field in result and result[field] is not None:
                result[field] = self.encrypt_field(field, result[field])

        if self._encrypt_user_id and "user_id" in result:
            result["user_id"] = self.encrypt_field("user_id", result["user_id"])
        if self._encrypt_session_id and "session_id" in result:
            result["session_id"] = self.encrypt_field("session_id", result["session_id"])

        return result

    def decrypt_memory_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Decrypt relevant fields in a memory row dict (after retrieval)."""
        if not self.is_enabled:
            return row

        result = dict(row)
        for field in self._encrypt_fields:
            if field in result and self._is_encrypted(result[field]):
                result[field] = self.decrypt_field(field, result[field])

        if self._encrypt_user_id and "user_id" in result and self._is_encrypted(result["user_id"]):
            result["user_id"] = self.decrypt_field("user_id", result["user_id"])
        if self._encrypt_session_id and "session_id" in result and self._is_encrypted(result["session_id"]):  # noqa: E501
            result["session_id"] = self.decrypt_field("session_id", result["session_id"])

        return result


# ---------------------------------------------------------------------------
# SQLCipher full-database encryption
# ---------------------------------------------------------------------------

class SQLCipherManager:
    """Manages SQLCipher connection configuration for SQLiteStorageAdapter.

    SQLCipher provides full-database AES-256 encryption at the SQLite level.
    The encryption is transparent to the application — SQL operations remain
    the same, but all data at rest is encrypted.

    Usage:
        manager = SQLCipherManager(key_file="/path/to/key")
        conn = manager.connect("kemi.db")
        # Use conn normally — all data is encrypted
    """

    def __init__(self, key: str | None = None, key_file: str | None = None) -> None:
        if key is None and key_file is None:
            raise EncryptionError("SQLCipher requires a key (key= or key_file=)")
        if key_file:
            key = load_key_from_file(key_file)
        self._key = key

    @property
    def key(self) -> str:
        return self._key

    def configure_connection(self, conn: Any) -> None:
        """Apply SQLCipher PRAGMAs to an existing sqlite3 connection.

        Must be called AFTER sqlite3.connect() but BEFORE any SQL operations.
        Sets the encryption key and cipher configuration.

        Uses hex-formatted key via PRAGMA key = "x'...'" to prevent any
        special-character issues in the PRAGMA value.
        """
        try:
            import sqlcipher3  # noqa: F401
        except ImportError as e:
            raise ConfigurationError(
                "sqlcipher3 package required for SQLCipher encryption. "
                "Install with: pip install kemi[sqlcipher] or pip install sqlcipher3"
            ) from e

        # Use hex-encoded key for safe PRAGMA key assignment without
        # any special-character injection risk.
        hex_key = self._key.encode("utf-8").hex()
        conn.execute(f'PRAGMA key = "x\'{hex_key}\'"')
        # Configure cipher settings for best security/compatibility
        conn.execute("PRAGMA cipher_page_size = 4096")
        conn.execute("PRAGMA kdf_iter = 256000")
        conn.execute("PRAGMA cipher_memory_security = ON")

    def connect(self, db_path: str) -> Any:
        """Create and return a SQLCipher-encrypted sqlite3 connection."""
        try:
            import sqlcipher3  # noqa: F401
        except ImportError as e:
            raise ConfigurationError(
                "sqlcipher3 package required for SQLCipher encryption. "
                "Install with: pip install kemi[sqlcipher] or pip install sqlcipher3"
            ) from e

        conn = sqlcipher3.connect(db_path)
        self.configure_connection(conn)
        return conn


def is_sqlcipher_available() -> bool:
    """Check if sqlcipher3 is installed and functional."""
    try:
        import sqlcipher3  # noqa: F401
        return True
    except ImportError:
        return False


def is_cryptography_available() -> bool:
    """Check if cryptography (Fernet) is installed and functional."""
    try:
        from cryptography.fernet import Fernet
        return True
    except ImportError:
        return False
