"""FastAPI REST API server for kemi memory.

Optional dependency: install with `pip install fastapi uvicorn`

Usage:
    from kemi.interfaces.api import create_app
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

Rate Limiting:
    Rate limiting is disabled by default. To enable, set:
    - KEMI_RATE_LIMIT_REQUESTS: max requests per window (default 100)
    - KEMI_RATE_LIMIT_WINDOW: time window in seconds (default 60)

Security:
    CORS and security headers are disabled by default. To enable, set:
    - KEMI_CORS_ORIGINS: comma-separated list of allowed origins

API Key Authentication (multi-tenancy):
    Disabled by default for backward compatibility. To enable, set:
    - KEMI_API_KEY_REQUIRED=true

    When the X-API-Key header is sent, the key is hashed with SHA-256 and
    looked up in the `api_keys` table. The associated user_id is injected
    into request.state.user_id. Endpoints then enforce that any user_id
    in the body/path matches the authenticated user, preventing cross-
    tenant access. Manage keys via the /api/keys endpoints or
    `kemi api-key ...` CLI commands.
"""

import logging
import os
import sqlite3
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from kemi import Memory
from kemi.exceptions import ConfigurationError, ValidationError
from kemi.infra.webhooks import (
    WebhookConfig,
    WebhookEventType,
    WebhookStore,
    validate_webhook_url,
)
from kemi.memory.model import LifecycleState, MemorySource, MemoryType

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False
    BaseModel = object

    def Field(*a: Any, **kw: Any) -> None:  # type: ignore[no-redef]  # noqa: N802
        return None


# Endpoints that never require auth, even when KEMI_API_KEY_REQUIRED=true.
# /health is a liveness probe; /api/keys POST bootstraps a brand-new user's
# first key (you can't authenticate before you have a key).
_AUTH_EXEMPT_PATHS = frozenset({"/health", "/api/keys"})

# Subset of /api/keys paths that are exempt: only the POST (create) and
# the health-style subpaths. GET/DELETE still require auth so that keys
# can't be enumerated by an anonymous caller.
_AUTH_EXEMPT_PREFIXES = tuple()  # handled per-route below


def _api_key_required() -> bool:
    """Whether the server requires X-API-Key on all non-exempt endpoints.

    Defaults to *True* when ``KEMI_API_KEYS_BOOTSTRAP`` is set (fail-closed
    for production), otherwise defaults to *False* for backward compatibility.
    """
    env = os.environ.get("KEMI_API_KEY_REQUIRED", "")
    if env:
        return env.lower() in ("true", "1", "yes")
    # Fail-closed default: require auth if bootstrap keys are configured.
    bootstrap = os.environ.get("KEMI_API_KEYS_BOOTSTRAP", "")
    return bootstrap.lower() in ("true", "1", "yes")


def _is_exempt(path: str, method: str) -> bool:
    """Whether (path, method) is exempt from API-key requirement."""
    if path in _AUTH_EXEMPT_PATHS:
        return True
    # POST /api/keys is the bootstrap endpoint; GET/DELETE are not.
    if path == "/api/keys" and method == "POST":
        return True
    return False


def _resolve_user_id(request: Request, claimed_user_id: str | None) -> str:
    """Return the effective user_id, enforcing isolation when authed.

    If a valid X-API-Key was presented, ``request.state.user_id`` is set.
    - When authed, ``claimed_user_id`` must match it (or be None); a
      mismatch raises 403 so a tenant cannot impersonate another.
    - When not authed, the caller's claimed_user_id is passed through.

    Returns the user_id that the endpoint should use.
    """
    authed = getattr(request.state, "user_id", None)
    if authed is None:
        if claimed_user_id is None:
            raise HTTPException(
                status_code=400,
                detail="user_id is required",
            )
        return claimed_user_id
    if claimed_user_id is not None and claimed_user_id != authed:
        raise HTTPException(
            status_code=403,
            detail="user_id does not match authenticated user",
        )
    return authed


def _require_admin(request: Request) -> str:
    """Require authentication; return the authed user_id.

    Used for endpoints that should only be reachable by an authenticated
    caller (e.g. listing other users' keys). The caller is responsible
    for any further authorization (e.g. role checks).
    """
    authed = getattr(request.state, "user_id", None)
    if authed is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return authed


class RateLimiter:
    """Simple in-memory rate limiter for API endpoints.

    Uses a sliding window approach with per-key counters.
    """

    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
    ) -> None:
        self._requests_per_window = requests_per_window
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if a request from the given key is allowed.

        Args:
            key: Identifier for the client (e.g., IP address, user_id).

        Returns:
            True if the request is allowed, False if rate limited.
        """
        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            # Clean up old timestamps
            self._requests[key] = [ts for ts in self._requests[key] if ts > window_start]

            if len(self._requests[key]) >= self._requests_per_window:
                return False

            self._requests[key].append(now)
            return True

    def get_retry_after(self, key: str) -> int:
        """Get seconds until the rate limit resets for this key."""
        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            valid_requests = [ts for ts in self._requests[key] if ts > window_start]
            if len(valid_requests) < self._requests_per_window:
                return 0

            oldest = min(valid_requests)
            return int(self._window_seconds - (now - oldest)) + 1


# Global rate limiter instance (lazily initialized)
_rate_limiter: RateLimiter | None = None


def _get_rate_limiter() -> RateLimiter | None:
    """Get or create the rate limiter based on environment config."""
    global _rate_limiter
    if _rate_limiter is not None:
        return _rate_limiter

    # Check if rate limiting is enabled via environment variables
    enabled = os.environ.get("KEMI_RATE_LIMIT_ENABLED", "false").lower() in ("true", "1", "yes")
    if not enabled:
        return None

    requests = int(os.environ.get("KEMI_RATE_LIMIT_REQUESTS", "100"))
    window = int(os.environ.get("KEMI_RATE_LIMIT_WINDOW", "60"))

    _rate_limiter = RateLimiter(requests_per_window=requests, window_seconds=window)
    logger.info(f"Rate limiting enabled: {requests} requests per {window} seconds")
    return _rate_limiter


def _check_rate_limit(client_key: str) -> tuple[bool, int]:
    """Check rate limit for a client key.

    Returns:
        Tuple of (is_allowed, retry_after_seconds).
        retry_after_seconds is 0 if allowed.
    """
    limiter = _get_rate_limiter()
    if limiter is None:
        return True, 0

    if limiter.is_allowed(client_key):
        return True, 0

    return False, limiter.get_retry_after(client_key)


# Cached APIKeyManager, lazily built from the active memory's storage.
_api_key_manager: Any = None
_api_key_manager_lock = Lock()


def _get_api_key_manager() -> Any:
    """Return a cached APIKeyManager bound to the active memory's storage.

    Returns None when the storage adapter doesn't support API keys
    (e.g. in-memory mock). Endpoints should treat None as 501.
    """
    global _api_key_manager
    if _api_key_manager is not None:
        return _api_key_manager
    with _api_key_manager_lock:
        if _api_key_manager is not None:
            return _api_key_manager
        mem = _get_memory_singleton()
        store = getattr(mem, "_store", None)
        # Prefer a real connection if the adapter exposes one; otherwise
        # fall back to the storage adapter's helper.
        get_mgr = getattr(store, "get_api_key_manager", None)
        if callable(get_mgr):
            _api_key_manager = get_mgr()
            return _api_key_manager
        # Last resort: try to share a SQLite connection. Mocks won't have one.
        get_conn = getattr(store, "_get_connection", None)
        if callable(get_conn):
            from kemi.infra.api_keys import APIKeyManager

            _api_key_manager = APIKeyManager(connection=get_conn())
            return _api_key_manager
        return None


def _reset_api_key_manager() -> None:
    """Clear the cached manager. Used by tests to swap storage backends."""
    global _api_key_manager
    with _api_key_manager_lock:
        _api_key_manager = None


class RememberRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    source: str = "user_stated"
    tags: list[str] | None = None
    namespace: str = "default"
    session_id: str | None = None
    memory_type: str = "episodic"
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class RecallRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1)
    max_tokens: int | None = None
    namespace: str = "default"
    session_id: str | None = None
    hybrid_search: bool | None = None


class UpdateRequest(BaseModel):
    content: str | None = None
    importance: float | None = Field(None, ge=0.0, le=1.0)
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    memory_type: str | None = None


class PruneRequest(BaseModel):
    max_age_days: float | None = None
    min_importance: float | None = None
    lifecycle_states: list[str] | None = None
    namespace: str = "default"


class ConsolidateRequest(BaseModel):
    namespace: str = "default"
    min_memories: int = 5
    max_age_days: float = 30.0


class TopicsRequest(BaseModel):
    n_clusters: int = 3
    namespace: str = "default"


class GraphRequest(BaseModel):
    namespace: str = "default"


class FeedbackRequest(BaseModel):
    memory_id: str
    helpful: bool = True
    namespace: str = "default"


class BatchRememberRequest(BaseModel):
    """Request for background batch remember operation."""

    user_id: str = Field(..., min_length=1)
    contents: list[str] = Field(..., min_length=1)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    namespace: str = "default"


class RebuildFTSRequest(BaseModel):
    """Request for background FTS index rebuild."""

    user_id: str | None = None  # Optional: rebuild for specific user only


class AdminFTSStatsRequest(BaseModel):
    """Request for FTS index statistics."""

    user_id: str | None = None  # Optional: get stats for specific user only


class AdminFTSRepairRequest(BaseModel):
    """Request for FTS index integrity repair."""

    verify_only: bool = False  # If True, only verify without repairing


class AuditLogRequest(BaseModel):
    """Request to log an audit entry."""

    user_id: str = Field(..., min_length=1)
    operation: str = Field(..., min_length=1)
    status: str = "success"
    details: dict[str, Any] | None = None
    memory_id: str | None = None
    namespace: str = "default"
    client_ip: str | None = None
    user_agent: str | None = None
    duration_ms: float | None = None


class AuditQueryRequest(BaseModel):
    """Request to query the audit trail."""

    user_id: str | None = None
    operation: str | None = None
    status: str | None = None
    memory_id: str | None = None
    namespace: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    limit: int = Field(100, ge=1, le=10000)
    offset: int = Field(0, ge=0)


class AuditExportRequest(BaseModel):
    """Request to export audit entries."""

    start_time: str | None = None
    end_time: str | None = None
    user_id: str | None = None


class AdaptiveAnalyzeRequest(BaseModel):
    """Request to analyze a query for adaptive retrieval."""

    query: str = Field(..., min_length=1)


class EnableFeatureRequest(BaseModel):
    """Request to enable or disable a feature."""

    enable: bool = True
    retention_days: int = Field(365, ge=1)
    auto_purge: bool = True


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key."""

    user_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=200)
    expires_in_days: int | None = Field(None, ge=1, le=36500)


# Global memory instance for lifespan management
_memory_instance: Memory | None = None


def _get_memory_singleton() -> Memory:
    """Get or create a singleton Memory instance."""
    global _memory_instance
    if _memory_instance is not None:
        return _memory_instance

    db_path = os.environ.get("KEMI_DB_PATH", os.path.expanduser("~/.kemi/memories.db"))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _memory_instance = Memory()
    return _memory_instance


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Lifespan context manager for graceful startup and shutdown.

    On startup: Initialize the memory instance
    On shutdown: Properly close database connections
    """
    # Startup
    logger.info("Starting kemi API server...")
    _get_memory_singleton()
    db_path = os.environ.get("KEMI_DB_PATH", "~/.kemi/memories.db")
    logger.info(f"Memory instance initialized with DB: {db_path}")

    # Security startup warnings
    if not _api_key_required():
        trusted = os.environ.get("KEMI_TRUSTED_HOSTS", "")
        if not trusted:
            logger.warning(
                "API key authentication is DISABLED and no KEMI_TRUSTED_HOSTS "
                "are set. The server is open to any caller. Set "
                "KEMI_API_KEY_REQUIRED=true or KEMI_TRUSTED_HOSTS to secure it."
            )

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down kemi API server...")
    global _memory_instance
    if _memory_instance is not None:
        # Close the storage adapter connection
        try:
            store = getattr(_memory_instance, "_store", None)
            if store is not None and hasattr(store, "close"):
                store.close()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
        _memory_instance = None

    logger.info("kemi API server shutdown complete")


def create_app(memory: Memory | None = None) -> Any:
    """Create a FastAPI application wrapping a kemi Memory instance.

    Args:
        memory: Optional pre-configured Memory instance.
                 If None, creates a default Memory.

    Returns:
        FastAPI app instance.
    """
    if not _FASTAPI_AVAILABLE:
        raise ConfigurationError(
            "FastAPI is required for the API server. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="kemi API",
        version="0.3.0",
        lifespan=lifespan,
    )

    # Configure CORS if origins are specified
    cors_origins = os.environ.get("KEMI_CORS_ORIGINS", "")
    if cors_origins:
        origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
        if origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                allow_headers=["Authorization", "Content-Type"],
            )
            logger.info(f"CORS enabled for origins: {origins}")

    # Add security headers middleware for production
    trusted_hosts = os.environ.get("KEMI_TRUSTED_HOSTS", "")
    if trusted_hosts:
        hosts = [h.strip() for h in trusted_hosts.split(",") if h.strip()]
        if hosts:
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
            logger.info(f"Trusted host middleware enabled for: {hosts}")

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next: Any) -> Any:
        """Validate X-API-Key and inject request.state.user_id.

        Behaviour:
        - If X-API-Key is provided, hash it and look it up. On success
          the key's user_id is attached to request.state. On failure
          (unknown / expired / revoked) return 401.
        - If X-API-Key is absent:
          - Endpoints in _AUTH_EXEMPT_PATHS are always allowed.
          - All other endpoints return 401 when KEMI_API_KEY_REQUIRED=true.
          - When KEMI_API_KEY_REQUIRED=false (default), the request
            proceeds unauthenticated for backward compatibility.
        """
        # Don't try to validate against a missing storage adapter
        # (e.g. in some test harnesses); behave as if no auth is available.
        header = request.headers.get("X-API-Key")
        path = request.url.path
        method = request.method

        if header:
            manager = _get_api_key_manager()
            if manager is None:
                return JSONResponse(
                    status_code=501,
                    content={"detail": "API key authentication not supported by this storage"},
                )
            key = manager.lookup(header)
            if key is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired API key"},
                )
            request.state.user_id = key.user_id
            request.state.api_key_id = key.key_id
        else:
            if _api_key_required() and not _is_exempt(path, method):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "X-API-Key header required"},
                )

        return await call_next(request)

    # Expose auth state on the request state for endpoints that want to
    # introspect it (e.g. /api/keys GET to scope listings).
    @app.middleware("http")
    async def _ensure_state_defaults(request: Request, call_next: Any) -> Any:
        if not hasattr(request.state, "user_id"):
            request.state.user_id = None
        if not hasattr(request.state, "api_key_id"):
            request.state.api_key_id = None
        return await call_next(request)

    mem = memory or _get_memory_singleton()

    @app.post("/remember")
    async def remember(req: RememberRequest, request: Request) -> dict[str, Any]:
        effective_user = _resolve_user_id(request, req.user_id)
        # Rate limit check
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=(
                f"Rate limit exceeded. Retry after {retry_after} seconds."
            ),
                headers={"Retry-After": str(retry_after)},
            )

        try:
            source = MemorySource(req.source)
            mtype = MemoryType(req.memory_type)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

        mid = mem.remember(
            user_id=effective_user,
            content=req.content,
            importance=req.importance,
            source=source,
            tags=req.tags,
            namespace=req.namespace,
            session_id=req.session_id,
            memory_type=mtype,
            confidence=req.confidence,
        )
        return {"memory_id": mid}

    @app.post("/recall")
    async def recall(req: RecallRequest, request: Request) -> dict[str, Any]:
        effective_user = _resolve_user_id(request, req.user_id)
        # Rate limit check
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        try:
            results = mem.recall(
                user_id=effective_user,
                query=req.query,
                top_k=req.top_k,
                max_tokens=req.max_tokens,
                namespace=req.namespace,
                session_id=req.session_id,
                hybrid_search=req.hybrid_search,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return {
            "results": [
                {
                    "memory_id": r.memory_id,
                    "content": r.content,
                    "score": r.score,
                    "importance": r.importance,
                    "lifecycle_state": r.lifecycle_state.value,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "tags": r.tags,
                    "memory_type": r.memory_type.value,
                    "confidence": r.confidence,
                    "session_id": r.session_id,
                    "namespace": r.namespace,
                    "version": r.version,
                }
                for r in results
            ]
        }

    @app.post("/recall/stream")
    async def recall_stream(req: RecallRequest, request: Request) -> StreamingResponse:
        """Stream recall results as Server-Sent Events."""
        import json

        effective_user = _resolve_user_id(request, req.user_id)

        # Rate limit check
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        async def _generate() -> Any:
            count = 0
            try:
                stream = mem.recall_stream(
                    user_id=effective_user,
                    query=req.query,
                    top_k=req.top_k,
                    max_tokens=req.max_tokens,
                    namespace=req.namespace,
                    session_id=req.session_id,
                    hybrid_search=req.hybrid_search,
                )
                async for result in stream:
                    count += 1
                    payload = {
                        "memory_id": result.memory_id,
                        "content": result.content,
                        "score": result.score,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            except ValueError as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                return

            yield f"event: done\ndata: {json.dumps({'total': count})}\n\n"

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
        )

    @app.post("/recall-explain")
    async def recall_explain(req: RecallRequest, request: Request) -> dict[str, Any]:
        effective_user = _resolve_user_id(request, req.user_id)
        # Rate limit check
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        try:
            explained = mem.recall_explain(
                user_id=effective_user,
                query=req.query,
                top_k=req.top_k,
                namespace=req.namespace,
                session_id=req.session_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return {
            "results": [
                {
                    "memory": {
                        "memory_id": item["memory"].memory_id,
                        "content": item["memory"].content,
                        "score": item["memory"].score,
                    },
                    "explanation": item["explanation"],
                }
                for item in explained
            ]
        }

    @app.post("/forget")
    async def forget(
        request: Request,
        user_id: str,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        effective_user = _resolve_user_id(request, user_id)
        # Rate limit check
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        count = mem.forget(effective_user, memory_id)
        return {"deleted": count}

    @app.patch("/memories/{memory_id}")
    async def update_memory(
        memory_id: str,
        req: UpdateRequest,
        request: Request,
    ) -> dict[str, Any]:
        # We need the user_id to rate-limit per-user; look up the memory.
        existing = mem._store.get(memory_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
        effective_user = _resolve_user_id(request, existing.user_id)
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        try:
            mtype = None
            if req.memory_type:
                mtype = MemoryType(req.memory_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        try:
            mem.update(
                memory_id=memory_id,
                content=req.content,
                importance=req.importance,
                confidence=req.confidence,
                memory_type=mtype,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"memory_id": memory_id, "status": "updated"}

    @app.post("/prune")
    async def prune(
        request: Request,
        user_id: str,
        req: PruneRequest,
    ) -> dict[str, Any]:
        effective_user = _resolve_user_id(request, user_id)
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        lifecycle_filter = None
        if req.lifecycle_states:
            try:
                lifecycle_filter = [LifecycleState(s) for s in req.lifecycle_states]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        deleted = mem.prune(
            user_id=effective_user,
            max_age_days=req.max_age_days,
            min_importance=req.min_importance,
            lifecycle_states=lifecycle_filter,
            namespace=req.namespace,
        )
        return {"deleted": deleted}

    @app.get("/stats/{user_id}")
    async def stats(user_id: str, request: Request) -> dict[str, Any]:
        effective_user = _resolve_user_id(request, user_id)
        # Rate limit check
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        try:
            return mem.stats(effective_user)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/users")
    async def list_users(request: Request) -> dict[str, Any]:
        # If authed, restrict to the caller's own user_id.
        authed = getattr(request.state, "user_id", None)
        users = mem.list_users()
        if authed is not None:
            users = [u for u in users if u == authed]
        return {"users": users}

    @app.post("/consolidate/{user_id}")
    async def consolidate_user(
        user_id: str,
        req: ConsolidateRequest,
        request: Request,
    ) -> dict[str, Any]:
        effective_user = _resolve_user_id(request, user_id)
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        try:
            mid = mem.consolidate(
                user_id=effective_user,
                namespace=req.namespace,
                min_memories=req.min_memories,
                max_age_days=req.max_age_days,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if mid:
            return {"consolidated_memory_id": mid}
        return {"message": "No consolidation needed"}

    @app.post("/topics/{user_id}")
    async def topics_user(
        user_id: str,
        req: TopicsRequest,
        request: Request,
    ) -> dict[str, Any]:
        effective_user = _resolve_user_id(request, user_id)
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        try:
            clusters = mem.cluster_topics(
                user_id=effective_user,
                n_clusters=req.n_clusters,
                namespace=req.namespace,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "topics": {
                label: [
                    {
                        "memory_id": m.memory_id,
                        "content": m.content,
                        "importance": m.importance,
                    }
                    for m in mems
                ]
                for label, mems in clusters.items()
            }
        }

    @app.post("/graph/{user_id}")
    async def graph_user(
        user_id: str,
        req: GraphRequest,
        request: Request,
    ) -> dict[str, Any]:
        effective_user = _resolve_user_id(request, user_id)
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        try:
            graph_data = mem.get_memory_graph(
                user_id=effective_user,
                namespace=req.namespace,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return graph_data

    @app.post("/feedback/{user_id}")
    async def feedback_user(
        user_id: str,
        req: FeedbackRequest,
        request: Request,
    ) -> dict[str, Any]:
        effective_user = _resolve_user_id(request, user_id)
        allowed, retry_after = _check_rate_limit(effective_user)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        try:
            mem.feedback(
                user_id=effective_user,
                memory_id=req.memory_id,
                helpful=req.helpful,
                namespace=req.namespace,
            )
        except (ValueError, AttributeError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "memory_id": req.memory_id, "helpful": req.helpful}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Enhanced health check endpoint.

        Returns:
            status: "ok" if healthy, "degraded" if issues detected
            components: Dictionary of component statuses
            timestamp: ISO timestamp of the health check
        """
        components: dict[str, Any] = {}
        overall_healthy = True
        mem = memory or _get_memory_singleton()

        # Check database connectivity
        try:
            if hasattr(mem, "_store") and mem._store:
                conn = mem._store._get_connection()
                cursor = conn.execute("SELECT 1")
                cursor.fetchone()
                components["database"] = {"status": "healthy", "type": "sqlite"}
            else:
                components["database"] = {"status": "unknown", "message": "Storage not initialized"}
                overall_healthy = False
        except (sqlite3.Error, AttributeError) as e:
            components["database"] = {"status": "unhealthy", "error": str(e)}
            overall_healthy = False

        # Check embedding adapter availability
        try:
            embed = getattr(mem, "_embed", None)
            if embed is not None:
                components["embedding"] = {"status": "healthy", "adapter": type(embed).__name__}
            else:
                components["embedding"] = {
                    "status": "not_configured",
                    "message": "No embedding adapter",
                }
        except AttributeError as e:
            components["embedding"] = {"status": "unhealthy", "error": str(e)}
            overall_healthy = False

        status = "ok" if overall_healthy else "degraded"

        return {
            "status": status,
            "components": components,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Background Task Endpoints

    @app.post("/tasks/embed-batch")
    async def submit_embed_batch_task(
        req: BatchRememberRequest, request: Request
    ) -> dict[str, Any]:
        """Submit a batch embedding task to run in background.

        This endpoint returns immediately with a task_id that can be used
        to check the task status at /tasks/{task_id}.

        Args:
            req: BatchRememberRequest with user_id, contents, etc.

        Returns:
            Dict with task_id for tracking progress.
        """
        effective_user = _resolve_user_id(request, req.user_id)
        from kemi.infra.background_tasks import get_task_manager

        task_manager = get_task_manager()
        try:
            task_id = task_manager.submit_embed_batch(
                user_id=effective_user,
                contents=req.contents,
                importance=req.importance,
                namespace=req.namespace,
                memory=mem,
            )
            return {"task_id": task_id, "status": "pending"}
        except RuntimeError as err:
            raise HTTPException(status_code=429, detail=str(err)) from err

    @app.post("/tasks/rebuild-fts")
    async def submit_rebuild_fts_task(
        req: RebuildFTSRequest, request: Request
    ) -> dict[str, Any]:
        """Submit an FTS index rebuild task to run in background.

        This endpoint returns immediately with a task_id that can be used
        to check the task status at /tasks/{task_id}.

        Args:
            req: RebuildFTSRequest with optional user_id filter.

        Returns:
            Dict with task_id for tracking progress.
        """
        # Optional user_id: when authed, we always scope to the authed user
        # regardless of what the body says (prevents cross-tenant rebuilds).
        authed = getattr(request.state, "user_id", None)
        target_user: str | None
        if authed is not None:
            if req.user_id is not None and req.user_id != authed:
                raise HTTPException(
                    status_code=403,
                    detail="user_id does not match authenticated user",
                )
            target_user = authed
        else:
            target_user = req.user_id

        from kemi.infra.background_tasks import get_task_manager

        task_manager = get_task_manager()
        try:
            task_id = task_manager.submit_rebuild_fts_index(user_id=target_user, memory=mem)
            return {"task_id": task_id, "status": "pending"}
        except RuntimeError as err:
            raise HTTPException(status_code=429, detail=str(err)) from err

    @app.get("/tasks/stats")
    async def get_task_stats() -> dict[str, Any]:
        """Get background task manager statistics.

        Returns:
            Dict with counts of pending, running, completed, failed tasks.
        """
        from kemi.infra.background_tasks import get_task_manager

        task_manager = get_task_manager()
        return task_manager.get_stats()

    @app.get("/tasks/{task_id}")
    async def get_task_status(task_id: str) -> dict[str, Any]:
        """Get the status of a background task.

        Args:
            task_id: The task ID returned from submit_* endpoints.

        Returns:
            Task status including progress, result, or error.
        """
        from kemi.infra.background_tasks import get_task_manager

        task_manager = get_task_manager()
        task = task_manager.get_task_status(task_id)

        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        return task.to_dict()

    @app.get("/tasks")
    async def list_tasks(
        status: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List all background tasks.

        Args:
            status: Optional filter by status (pending, running, completed, failed).
            limit: Maximum number of tasks to return (default 50).

        Returns:
            List of tasks with their statuses.
        """
        from kemi.infra.background_tasks import TaskStatus, get_task_manager

        task_manager = get_task_manager()

        filter_status = None
        if status:
            try:
                filter_status = TaskStatus(status)
            except ValueError as exc:
                valid_vals = "pending, running, completed, failed"
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid values: {valid_vals}",
                ) from exc

        tasks = task_manager.list_tasks(status=filter_status, limit=limit)
        return {
            "tasks": [t.to_dict() for t in tasks],
            "stats": task_manager.get_stats(),
        }

    @app.delete("/tasks/{task_id}")
    async def cancel_task(task_id: str) -> dict[str, Any]:
        """Cancel a pending background task.

        Note: Running tasks cannot be cancelled mid-execution.

        Args:
            task_id: The task ID to cancel.

        Returns:
            Dict with success status.
        """
        from kemi.infra.background_tasks import get_task_manager

        task_manager = get_task_manager()
        cancelled = task_manager.cancel_task(task_id)

        if not cancelled:
            raise HTTPException(
                status_code=400,
                detail="Cannot cancel task: not found or already running",
            )

        return {"task_id": task_id, "cancelled": True}

    # Admin Endpoints for Index Maintenance

    @app.post("/admin/fts/rebuild")
    async def admin_rebuild_fts(request: Request) -> dict[str, Any]:
        """Admin endpoint to rebuild FTS5 index synchronously.

        This is a blocking operation that rebuilds the full-text search index.
        For large datasets, consider using the background task endpoint instead.

        Returns:
            Dict with rebuild statistics.
        """
        _require_admin(request)
        allowed, retry_after = _check_rate_limit("admin:fts:rebuild")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        mem = _get_memory_singleton()

        if not hasattr(mem._store, "rebuild_fts_index"):
            raise HTTPException(
                status_code=501,
                detail="Storage adapter does not support FTS index rebuild",
            )

        try:
            # Rebuild entire FTS index
            count = mem._store.rebuild_fts_index()
            return {
                "status": "completed",
                "memories_indexed": count,
                "scope": "all",
            }
        except Exception as e:
            logger.error(f"FTS rebuild failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/admin/fts/stats")
    async def admin_fts_stats(
        request: Request, user_id: str | None = None
    ) -> dict[str, Any]:
        """Admin endpoint to get FTS5 index statistics.

        Args:
            user_id: Optional user ID to get stats for specific user.

        Returns:
            Dict with FTS index statistics.
        """
        _require_admin(request)
        allowed, retry_after = _check_rate_limit("admin:fts:stats")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        # If authed, restrict the stats view to the caller's own user.
        if request is not None:
            authed = getattr(request.state, "user_id", None)
            if authed is not None:
                if user_id is not None and user_id != authed:
                    raise HTTPException(
                        status_code=403,
                        detail="user_id does not match authenticated user",
                    )
                user_id = authed

        mem = _get_memory_singleton()

        try:
            conn = mem._store._get_connection()

            # Get total FTS entries
            cursor = conn.execute("SELECT COUNT(*) FROM memories_fts")
            fts_total = cursor.fetchone()[0]

            # Get total memories in main table
            if user_id:
                cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,))
                mem_total = cursor.fetchone()[0]

                # Get FTS entries for this user
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM memories_fts WHERE user_id = ?", (user_id,)
                )
                fts_user = cursor.fetchone()[0]

                in_sync = (fts_user == mem_total) if mem_total > 0 else True

                return {
                    "fts_total_entries": fts_total,
                    "user_id": user_id,
                    "user_memories": mem_total,
                    "user_fts_entries": fts_user,
                    "in_sync": in_sync,
                    "sync_gap": mem_total - fts_user,
                }
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
                mem_total = cursor.fetchone()[0]

                in_sync = (fts_total == mem_total) if mem_total > 0 else True

                return {
                    "fts_total_entries": fts_total,
                    "total_memories": mem_total,
                    "in_sync": in_sync,
                    "sync_gap": mem_total - fts_total,
                }
        except Exception as e:
            logger.error(f"FTS stats failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/admin/fts/verify")
    async def admin_fts_verify(req: AdminFTSRepairRequest, request: Request) -> dict[str, Any]:
        """Admin endpoint to verify FTS5 index integrity.

        Checks if all memories have corresponding FTS entries and vice versa.

        Args:
            req: AdminFTSRepairRequest with verify_only flag.

        Returns:
            Dict with verification results.
        """
        _require_admin(request)
        allowed, retry_after = _check_rate_limit("admin:fts:verify")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        mem = _get_memory_singleton()

        try:
            conn = mem._store._get_connection()

            # Get all memory IDs
            cursor = conn.execute("SELECT memory_id FROM memories")
            memory_ids = set(row[0] for row in cursor.fetchall())

            # Get all FTS IDs
            cursor = conn.execute("SELECT memory_id FROM memories_fts")
            fts_ids = set(row[0] for row in cursor.fetchall())

            # Find discrepancies
            missing_from_fts = memory_ids - fts_ids
            orphaned_in_fts = fts_ids - memory_ids

            in_sync = len(missing_from_fts) == 0 and len(orphaned_in_fts) == 0

            result = {
                "status": "ok" if in_sync else "degraded",
                "total_memories": len(memory_ids),
                "total_fts_entries": len(fts_ids),
                "in_sync": in_sync,
                "missing_from_fts": len(missing_from_fts),
                "orphaned_in_fts": len(orphaned_in_fts),
            }

            if not in_sync and not req.verify_only:
                # Auto-repair: remove orphaned FTS entries
                if orphaned_in_fts:
                    placeholders = ",".join("?" * len(orphaned_in_fts))
                    conn.execute(
                        f"DELETE FROM memories_fts WHERE memory_id IN ({placeholders})",
                        list(orphaned_in_fts),
                    )
                    result["repaired_orphaned"] = len(orphaned_in_fts)

                result["auto_repaired"] = True
                result["status"] = "repaired"

            return result

        except Exception as e:
            logger.error(f"FTS verify failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/admin/health")
    async def admin_health() -> dict[str, Any]:
        """Admin health check endpoint with detailed system status.

        Returns:
            Dict with detailed component statuses and system metrics.
        """
        mem = _get_memory_singleton()
        components: dict[str, Any] = {}

        # Database health
        try:
            conn = mem._store._get_connection()
            cursor = conn.execute("SELECT 1")
            cursor.fetchone()

            # Get database stats
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            total_memories = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM memories_fts")
            fts_entries = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(DISTINCT user_id) FROM memories")
            total_users = cursor.fetchone()[0]

            components["database"] = {
                "status": "healthy",
                "type": "sqlite",
                "total_memories": total_memories,
                "total_users": total_users,
                "fts_entries": fts_entries,
                "fts_in_sync": total_memories == fts_entries,
            }
        except Exception as e:
            components["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Embedding adapter health
        try:
            embed = getattr(mem, "_embed", None)
            if embed is not None:
                adapter_name = type(embed).__name__
                components["embedding"] = {
                    "status": "healthy",
                    "adapter": adapter_name,
                }

                # Check circuit breaker if available
                if hasattr(embed, "get_circuit_breaker_state"):
                    cb_state = embed.get_circuit_breaker_state()
                    components["embedding"]["circuit_breaker"] = cb_state
            else:
                components["embedding"] = {
                    "status": "not_configured",
                }
        except Exception as e:
            components["embedding"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Background task manager health
        try:
            from kemi.infra.background_tasks import get_task_manager

            task_manager = get_task_manager()
            stats = task_manager.get_stats()
            components["task_manager"] = {
                "status": "healthy",
                "pending": stats["pending"],
                "running": stats["running"],
                "completed": stats["completed"],
                "failed": stats["failed"],
            }
        except Exception as e:
            components["task_manager"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Determine overall status
        all_healthy = all(c.get("status") == "healthy" for c in components.values())


        return {
            "status": "ok" if all_healthy else "degraded",
            "components": components,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Observability / Metrics Endpoints

    @app.get("/metrics")
    async def get_metrics(output_format: str = "json") -> Any:
        """Get system metrics.

        Args:
            format: Output format — "json" or "prometheus".

        Returns:
            Metrics data in the requested format.
        """
        metrics_data = mem.get_metrics()
        if metrics_data is None:
            raise HTTPException(
                status_code=503,
                detail="Metrics collector not available",
            )

        if output_format.lower() == "prometheus":
            prom = mem.get_metrics_prometheus()
            if prom is None:
                raise HTTPException(
                    status_code=503,
                    detail="Metrics collector not available",
                )
            return PlainTextResponse(content=prom, media_type="text/plain")

        return metrics_data

    # Audit Trail Endpoints

    @app.post("/audit/log")
    async def audit_log(req: AuditLogRequest, request: Request) -> dict[str, Any]:
        """Log an operation to the audit trail.

        Returns:
            Dict with the entry ID of the logged operation.
        """
        effective_user = _resolve_user_id(request, req.user_id)
        if not hasattr(mem, "_audit_trail") or mem._audit_trail is None:
            raise HTTPException(
                status_code=503,
                detail="Audit trail not enabled. Use POST /admin/enable-audit first.",
            )

        try:
            entry_id = mem._audit_trail.log_operation(
                user_id=effective_user,
                operation=req.operation,
                details=req.details or {},
                memory_id=req.memory_id,
                namespace=req.namespace,
                status=req.status,
                client_ip=req.client_ip,
                user_agent=req.user_agent,
                duration_ms=req.duration_ms,
            )
            return {"entry_id": entry_id, "status": "logged"}
        except Exception as e:
            logger.error(f"Audit log failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/audit/query")
    async def audit_query(
        req: AuditQueryRequest, request: Request
    ) -> dict[str, Any]:
        """Query the audit trail with filters.

        Returns:
            Dict with list of matching audit entries and total count.
        """
        if not hasattr(mem, "_audit_trail") or mem._audit_trail is None:
            raise HTTPException(
                status_code=503,
                detail="Audit trail not enabled. Use POST /admin/enable-audit first.",
            )

        # When authed, force the user_id filter to the caller's own id
        # so a tenant cannot read another tenant's audit entries.
        authed = getattr(request.state, "user_id", None)
        query_user = req.user_id
        if authed is not None:
            if query_user is not None and query_user != authed:
                raise HTTPException(
                    status_code=403,
                    detail="user_id does not match authenticated user",
                )
            query_user = authed

        try:
            entries = mem._audit_trail.query(
                user_id=query_user,
                operation=req.operation,
                status=req.status,
                memory_id=req.memory_id,
                namespace=req.namespace,
                start_time=req.start_time,
                end_time=req.end_time,
                limit=req.limit,
                offset=req.offset,
            )
            return {
                "entries": [e.to_dict() for e in entries],
                "count": len(entries),
                "limit": req.limit,
                "offset": req.offset,
            }
        except Exception as e:
            logger.error(f"Audit query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/audit/stats")
    async def audit_stats() -> dict[str, Any]:
        """Get overall audit trail statistics.

        Returns:
            Dict with total entries, unique users, date range, retention policy.
        """
        if not hasattr(mem, "_audit_trail") or mem._audit_trail is None:
            raise HTTPException(
                status_code=503,
                detail="Audit trail not enabled. Use POST /admin/enable-audit first.",
            )

        try:
            return mem._audit_trail.get_stats()  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Audit stats failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/audit/export")
    async def audit_export(
        req: AuditExportRequest, request: Request
    ) -> dict[str, Any]:
        """Export audit entries for compliance.

        Returns:
            Dict with exported entries.
        """
        if not hasattr(mem, "_audit_trail") or mem._audit_trail is None:
            raise HTTPException(
                status_code=503,
                detail="Audit trail not enabled. Use POST /admin/enable-audit first.",
            )

        # Same isolation rule as audit_query.
        authed = getattr(request.state, "user_id", None)
        export_user = req.user_id
        if authed is not None:
            if export_user is not None and export_user != authed:
                raise HTTPException(
                    status_code=403,
                    detail="user_id does not match authenticated user",
                )
            export_user = authed

        try:
            entries = mem._audit_trail.export(
                start_time=req.start_time,
                end_time=req.end_time,
                user_id=export_user,
            )
            return {"entries": entries, "count": len(entries)}
        except Exception as e:
            logger.error(f"Audit export failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Adaptive Retrieval Endpoints

    @app.post("/adaptive/analyze")
    async def adaptive_analyze(req: AdaptiveAnalyzeRequest) -> dict[str, Any]:
        """Analyze a query and return adaptive retrieval weights.

        Returns:
            Dict with query classification, confidence, and recommended weights.
        """
        if not hasattr(mem, "_adaptive_retriever") or mem._adaptive_retriever is None:
            raise HTTPException(
                status_code=503,
                detail="Adaptive retrieval not enabled. Use POST /admin/enable-adaptive first.",
            )

        try:
            profile = mem._adaptive_retriever.analyze_query(req.query)
            return {
                "query": profile.query,
                "query_type": profile.query_type.value,
                "confidence": profile.confidence,
                "word_count": profile.word_count,
                "keyword_density": profile.keyword_density,
                "specificity": profile.specificity,
                "has_question_mark": profile.has_question_mark,
                "has_named_entity_hint": profile.has_named_entity_hint,
                "recommended_weights": profile.recommended_weights,
            }
        except Exception as e:
            logger.error(f"Adaptive analyze failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/adaptive/user-profile/{user_id}")
    async def adaptive_user_profile(
        user_id: str, request: Request
    ) -> dict[str, Any]:
        """Get the adaptive query type distribution for a user.

        Returns:
            Dict with query distribution and dominant type.
        """
        effective_user = _resolve_user_id(request, user_id)
        if not hasattr(mem, "_adaptive_retriever") or mem._adaptive_retriever is None:
            raise HTTPException(
                status_code=503,
                detail="Adaptive retrieval not enabled. Use POST /admin/enable-adaptive first.",
            )

        try:
            return mem._adaptive_retriever.get_user_profile(effective_user)  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Adaptive user profile failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Admin Feature Toggle Endpoints

    @app.post("/admin/enable-audit")
    async def admin_enable_audit(req: EnableFeatureRequest) -> dict[str, Any]:
        """Enable or disable the audit trail.

        Returns:
            Dict with enabled status.
        """
        if not hasattr(mem, "enable_audit_trail"):
            raise HTTPException(
                status_code=501,
                detail="Memory instance does not support audit trail",
            )

        try:
            if req.enable:
                mem.enable_audit_trail(
                    retention_days=req.retention_days,
                    auto_purge=req.auto_purge,
                )
            else:
                mem._audit_trail = None

            return {
                "audit_trail_enabled": req.enable,
                "retention_days": req.retention_days,
                "auto_purge": req.auto_purge,
            }
        except Exception as e:
            logger.error(f"Enable audit trail failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/admin/enable-adaptive")
    async def admin_enable_adaptive(req: EnableFeatureRequest) -> dict[str, Any]:
        """Enable or disable adaptive retrieval.

        Returns:
            Dict with enabled status.
        """
        if not hasattr(mem, "enable_adaptive_retrieval"):
            raise HTTPException(
                status_code=501,
                detail="Memory instance does not support adaptive retrieval",
            )

        try:
            mem.enable_adaptive_retrieval(enable=req.enable)
            return {"adaptive_retrieval_enabled": req.enable}
        except Exception as e:
            logger.error(f"Enable adaptive retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # API Key Management Endpoints

    @app.post("/api/keys")
    async def create_api_key(
        req: CreateAPIKeyRequest, request: Request
    ) -> dict[str, Any]:
        """Create a new API key for a user.

        The raw key is returned in the response exactly once; it cannot
        be retrieved later. Store it securely.

        When the caller is authenticated, the key is bound to the
        caller's own user_id (a 403 is raised if the body disagrees).
        When unauthenticated, the body's user_id is used — this is the
        bootstrap path for a brand-new tenant.
        """
        manager = _get_api_key_manager()
        if manager is None:
            raise HTTPException(
                status_code=501,
                detail="API key management not supported by this storage",
            )

        authed = getattr(request.state, "user_id", None)
        if authed is not None and req.user_id != authed:
            raise HTTPException(
                status_code=403,
                detail="user_id does not match authenticated user",
            )

        from kemi.infra.api_keys import make_expiry

        expires_at = make_expiry(req.expires_in_days) if req.expires_in_days else None
        try:
            key = manager.create_key(
                user_id=req.user_id,
                name=req.name,
                expires_at=expires_at,
            )
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return key.to_dict(include_secret=True)

    @app.get("/api/keys")
    async def list_api_keys(request: Request) -> dict[str, Any]:
        """List API keys. Authenticated callers see only their own keys."""
        manager = _get_api_key_manager()
        if manager is None:
            raise HTTPException(
                status_code=501,
                detail="API key management not supported by this storage",
            )

        authed = getattr(request.state, "user_id", None)
        # When authed, always scope to the caller's user_id regardless of
        # any user_id query param. When unauthed, the listing is global
        # (admin-style view).
        scope_user = authed
        keys = manager.list_keys(user_id=scope_user)
        return {
            "keys": [k.to_dict() for k in keys],
            "count": len(keys),
        }

    @app.delete("/api/keys/{key_id}")
    async def revoke_api_key(key_id: str, request: Request) -> dict[str, Any]:
        """Revoke an API key by id.

        Authenticated callers may only revoke their own keys.
        """
        manager = _get_api_key_manager()
        if manager is None:
            raise HTTPException(
                status_code=501,
                detail="API key management not supported by this storage",
            )

        authed = getattr(request.state, "user_id", None)
        if authed is not None:
            existing = manager.get(key_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="Key not found")
            if existing.user_id != authed:
                raise HTTPException(
                    status_code=403,
                    detail="Cannot revoke a key belonging to another user",
                )

        if not manager.revoke(key_id):
            raise HTTPException(
                status_code=404,
                detail="Key not found or already revoked",
            )
        return {"key_id": key_id, "revoked": True}

    # Memory Version History Endpoint

    @app.get("/memories/{memory_id}/history")
    async def get_memory_history(memory_id: str, request: Request, limit: int = 100) -> dict[str, Any]:  # noqa: E501
        """Get version history for a memory."""
        existing = mem._store.get(memory_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
        # History contains potentially sensitive snapshot data; require auth.
        authed = getattr(request.state, "user_id", None)
        if authed is None and _api_key_required():
            raise HTTPException(status_code=401, detail="Authentication required")
        _resolve_user_id(request, existing.user_id)
        try:
            mem.configure_versioning()
            history = mem.get_history(memory_id, limit=limit)
        except RuntimeError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e

        return {
            "memory_id": memory_id,
            "versions": [
                {
                    "version": snap.version,
                    "content": snap.content,
                    "importance": snap.importance,
                    "tags": snap.tags,
                    "memory_type": snap.memory_type,
                    "confidence": snap.confidence,
                    "namespace": snap.namespace,
                    "source": snap.source,
                    "changed_at": snap.changed_at.isoformat() if snap.changed_at else None,
                    "changed_by": snap.changed_by,
                }
                for snap in history
            ],
            "count": len(history),
        }

    # Webhook Management Endpoints

    class CreateWebhookRequest(BaseModel):
        url: str = Field(..., min_length=1)
        events: list[str] = Field(..., min_length=1)
        secret: str = ""
        active: bool = True

    class UpdateWebhookRequest(BaseModel):
        url: str | None = None
        events: list[str] | None = None
        secret: str | None = None
        active: bool | None = None

    def _get_webhook_store() -> WebhookStore | None:
        """Get or create a WebhookStore bound to the active memory's database."""
        try:
            db_path = mem._store._db_path
            return WebhookStore(db_path=db_path)
        except (AttributeError, Exception):
            return None

    @app.post("/webhooks", status_code=201)
    async def create_webhook(req: CreateWebhookRequest, request: Request) -> dict[str, Any]:
        """Register a new webhook endpoint."""
        store = _get_webhook_store()
        if store is None:
            raise HTTPException(
                status_code=501,
                detail="Webhook store not available (storage adapter does not expose db_path)",
            )

        try:
            event_types = [WebhookEventType.from_string(e) for e in req.events]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        try:
            validate_webhook_url(req.url)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        cfg = WebhookConfig(
            webhook_id="",
            url=req.url,
            events=event_types,
            secret=req.secret,
            active=req.active,
        )
        wh_id = store.create(cfg)
        return {"webhook_id": wh_id, "url": req.url, "events": req.events, "active": req.active}

    @app.get("/webhooks")
    async def list_webhooks(request: Request) -> dict[str, Any]:
        """List all registered webhooks."""
        store = _get_webhook_store()
        if store is None:
            raise HTTPException(status_code=501, detail="Webhook store not available")

        configs = store.list_all(active_only=False)
        return {
            "webhooks": [
                {
                    "webhook_id": c.webhook_id,
                    "url": c.url,
                    "events": [e.value for e in c.events],
                    "active": c.active,
                }
                for c in configs
            ],
            "count": len(configs),
        }

    @app.delete("/webhooks/{webhook_id}")
    async def delete_webhook(webhook_id: str, request: Request) -> dict[str, Any]:
        """Delete a webhook configuration."""
        store = _get_webhook_store()
        if store is None:
            raise HTTPException(status_code=501, detail="Webhook store not available")

        if not store.delete(webhook_id):
            raise HTTPException(status_code=404, detail=f"Webhook not found: {webhook_id}")
        return {"webhook_id": webhook_id, "deleted": True}

    # Admin endpoint: list users with their memory counts

    @app.get("/admin/users")
    async def admin_list_users(request: Request) -> dict[str, Any]:
        """List all users and their memory counts.

        When authenticated, only the caller's own row is returned.
        When unauthenticated (default in backward-compat mode), the
        full list is returned.
        """
        authed = getattr(request.state, "user_id", None)
        all_users = mem.list_users()
        if authed is not None:
            users = [u for u in all_users if u == authed]
        else:
            users = all_users

        store = getattr(mem, "_store", None)
        rows: list[dict[str, Any]] = []

        def _build_user_row(uid: str) -> dict[str, Any]:
            try:
                count = store.count(uid) if store is not None else 0
            except Exception:  # pragma: no cover - defensive
                count = 0
            row: dict[str, Any] = {"user_id": uid, "memory_count": count}
            last_active = getattr(store, "get_last_active", lambda _u: None)(uid)
            if last_active is not None:
                row["last_active"] = last_active
            return row

        # Fast-path: if the store exposes a raw SQLite connection, use a
        # single GROUP BY query instead of N+1 count() calls.
        get_conn = getattr(store, "_get_connection", None)
        if callable(get_conn) and users:
            try:
                conn = get_conn()
                placeholders = ",".join("?" * len(users))
                cursor = conn.execute(
                    f"SELECT user_id, COUNT(*) FROM memories WHERE user_id IN ({placeholders}) GROUP BY user_id",  # noqa: E501
                    users,
                )
                count_map = {row[0]: row[1] for row in cursor.fetchall()}
                for uid in users:
                    count = count_map.get(uid, 0)
                    row = {"user_id": uid, "memory_count": count}
                    last_active = getattr(store, "get_last_active", lambda _u: None)(uid)
                    if last_active is not None:
                        row["last_active"] = last_active
                    rows.append(row)
            except Exception:  # pragma: no cover - defensive
                # Fall back to the slow per-user loop on any error.
                rows = [_build_user_row(uid) for uid in users]
        else:
            rows = [_build_user_row(uid) for uid in users]
        return {"users": rows, "count": len(rows)}

    return app
