"""Custom exception hierarchy for the kemi library."""

from __future__ import annotations

from typing import Any

__all__ = [
    "KemiError",
    "ConfigurationError",
    "ValidationError",
    "NotFoundError",
    "EmbeddingError",
    "StorageError",
    "MigrationError",
    "IncompatibleSchemaError",
    "EncryptionError",
    "CompatibilityError",
]


class KemiError(Exception):
    """Base exception for all kemi errors."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context)

    def __str__(self) -> str:
        if not self.context:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} ({rendered})"


class ConfigurationError(KemiError):
    """Raised when a Memory or adapter is misconfigured."""


class ValidationError(ValueError, KemiError):
    """Raised when input fails validation (e.g. empty user_id)."""


class NotFoundError(LookupError, KemiError):
    """Raised when a requested record does not exist."""


class EmbeddingError(RuntimeError, KemiError):
    """Raised when an embedding operation fails."""


class StorageError(OSError, KemiError):
    """Raised when a storage operation fails."""

    def __init__(self, message: str, **context: Any) -> None:
        KemiError.__init__(self, message, **context)
        OSError.__init__(self, message)


class MigrationError(RuntimeError, KemiError):
    """Raised when schema migration or re-embedding fails."""


class IncompatibleSchemaError(RuntimeError, KemiError):
    """Raised when the stored schema version is incompatible."""


class EncryptionError(RuntimeError, KemiError):
    """Raised when encryption or decryption fails."""


class CompatibilityError(RuntimeError, KemiError):
    """Raised when a plugin was built for an incompatible protocol version."""
