"""HTTP API server (``kemi.interfaces.api`` re-exports the FastAPI
application factory and request models).
"""

from kemi.interfaces.api.app import (
    AuditQueryRequest,
    BatchRememberRequest,
    ConsolidateRequest,
    CreateAPIKeyRequest,
    PruneRequest,
    RateLimiter,
    RecallRequest,
    RememberRequest,
    UpdateRequest,
    _require_admin,
    _reset_api_key_manager,
    _resolve_user_id,
    create_app,
)

__all__ = [
    "AuditQueryRequest",
    "BatchRememberRequest",
    "ConsolidateRequest",
    "CreateAPIKeyRequest",
    "PruneRequest",
    "RateLimiter",
    "RecallRequest",
    "RememberRequest",
    "UpdateRequest",
    "_require_admin",
    "_reset_api_key_manager",
    "_resolve_user_id",
    "create_app",
]

