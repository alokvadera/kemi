"""Tests for the webhook callback system."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from kemi.exceptions import ValidationError
from kemi.infra.webhooks import (
    RetryConfig,
    WebhookConfig,
    WebhookDispatcher,
    WebhookEventType,
    WebhookStore,
    build_payload,
    sign_payload,
    validate_webhook_url,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path() -> str:
    """Provide a temporary SQLite database path for webhook store tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def store(db_path: str) -> WebhookStore:
    return WebhookStore(db_path=db_path)


@pytest.fixture
def sample_config() -> WebhookConfig:
    return WebhookConfig(
        webhook_id="",
        url="https://example.com/hook",
        events=[WebhookEventType.REMEMBERED, WebhookEventType.UPDATED],
        secret="test-secret",
        active=True,
    )


# ---------------------------------------------------------------------------
# SSRF URL validation
# ---------------------------------------------------------------------------


class TestValidateWebhookUrl:
    def test_valid_http_url(self) -> None:
        validate_webhook_url("http://example.com/hook")

    def test_valid_https_url(self) -> None:
        validate_webhook_url("https://example.com/hook")

    def test_valid_https_with_port(self) -> None:
        validate_webhook_url("https://example.com:8443/hook")

    def test_rejects_ftp_scheme(self) -> None:
        with pytest.raises(ValidationError, match="scheme must be http or https"):
            validate_webhook_url("ftp://example.com/hook")

    def test_rejects_file_scheme(self) -> None:
        with pytest.raises(ValidationError, match="scheme must be http or https"):
            validate_webhook_url("file:///etc/passwd")

    def test_rejects_javascript_scheme(self) -> None:
        with pytest.raises(ValidationError, match="scheme must be http or https"):
            validate_webhook_url("javascript:alert(1)")

    def test_rejects_localhost(self) -> None:
        with pytest.raises(ValidationError, match="localhost"):
            validate_webhook_url("http://localhost/hook")

    def test_rejects_localhost_case_insensitive(self) -> None:
        with pytest.raises(ValidationError, match="localhost"):
            validate_webhook_url("http://LOCALHOST/hook")

    def test_rejects_127_0_0_1(self) -> None:
        with pytest.raises(ValidationError, match="localhost"):
            validate_webhook_url("http://127.0.0.1/hook")

    def test_rejects_0_0_0_0(self) -> None:
        with pytest.raises(ValidationError, match="localhost"):
            validate_webhook_url("http://0.0.0.0/hook")

    def test_rejects_loopback_ipv6(self) -> None:
        with pytest.raises(ValidationError, match="localhost"):
            validate_webhook_url("http://[::1]/hook")

    def test_rejects_private_10_x(self) -> None:
        with pytest.raises(ValidationError, match="private or reserved"):
            validate_webhook_url("http://10.0.0.1/hook")

    def test_rejects_private_192_168_x(self) -> None:
        with pytest.raises(ValidationError, match="private or reserved"):
            validate_webhook_url("http://192.168.1.1/hook")

    def test_rejects_private_172_16_x(self) -> None:
        with pytest.raises(ValidationError, match="private or reserved"):
            validate_webhook_url("http://172.16.0.1/hook")

    def test_rejects_link_local_169_254(self) -> None:
        with pytest.raises(ValidationError, match="private or reserved"):
            validate_webhook_url("http://169.254.1.1/hook")

    def test_rejects_link_local_ipv6(self) -> None:
        with pytest.raises(ValidationError, match="private or reserved"):
            validate_webhook_url("http://[fe80::1]/hook")

    def test_rejects_unspecified_ipv4(self) -> None:
        with pytest.raises(ValidationError, match="localhost"):
            validate_webhook_url("http://0.0.0.0/hook")

    def test_rejects_unspecified_ipv6(self) -> None:
        with pytest.raises(ValidationError, match="private or reserved"):
            validate_webhook_url("http://[::]/hook")

    def test_rejects_multicast(self) -> None:
        with pytest.raises(ValidationError, match="private or reserved"):
            validate_webhook_url("http://224.0.0.1/hook")

    def test_rejects_missing_hostname(self) -> None:
        with pytest.raises(ValidationError, match="valid hostname"):
            validate_webhook_url("http:///hook")

    def test_rejects_reserved_ipv4(self) -> None:
        with pytest.raises(ValidationError, match="private or reserved"):
            validate_webhook_url("http://240.0.0.1/hook")

    def test_public_ip_allowed(self) -> None:
        validate_webhook_url("http://8.8.8.8/hook")

    def test_public_ipv6_allowed(self) -> None:
        validate_webhook_url("http://[2001:4860:4860::8888]/hook")


# ---------------------------------------------------------------------------
# WebhookEventType
# ---------------------------------------------------------------------------


class TestWebhookEventType:
    def test_from_string_valid(self) -> None:
        t = WebhookEventType.from_string("memory.remembered")
        assert t == WebhookEventType.REMEMBERED

    def test_from_string_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid event type"):
            WebhookEventType.from_string("invalid.event")

    def test_from_string_case_sensitive(self) -> None:
        with pytest.raises(ValueError):
            WebhookEventType.from_string("MEMORY.REMEMBERED")


# ---------------------------------------------------------------------------
# RetryConfig
# ---------------------------------------------------------------------------


class TestRetryConfig:
    def test_default_values(self) -> None:
        r = RetryConfig()
        assert r.max_retries == 5
        assert r.base_delay_seconds == 1.0
        assert r.max_delay_seconds == 60.0
        assert r.backoff_multiplier == 2.0

    def test_delay_calculation(self) -> None:
        r = RetryConfig(base_delay_seconds=1.0, max_delay_seconds=10.0)
        assert r.delay(0) == 1.0
        assert r.delay(1) == 2.0
        assert r.delay(2) == 4.0
        assert r.delay(3) == 8.0
        # Clamped to max_delay
        assert r.delay(4) == 10.0


# ---------------------------------------------------------------------------
# WebhookConfig
# ---------------------------------------------------------------------------


class TestWebhookConfig:
    def test_matches_event_active(self, sample_config: WebhookConfig) -> None:
        assert sample_config.matches_event(WebhookEventType.REMEMBERED)
        assert sample_config.matches_event(WebhookEventType.UPDATED)
        assert not sample_config.matches_event(WebhookEventType.DELETED)

    def test_matches_event_inactive(self, sample_config: WebhookConfig) -> None:
        sample_config.active = False
        assert not sample_config.matches_event(WebhookEventType.REMEMBERED)
        assert not sample_config.matches_event(WebhookEventType.UPDATED)

    def test_matches_event_empty_events(self) -> None:
        cfg = WebhookConfig(webhook_id="test", url="http://example.com", events=[])
        assert not cfg.matches_event(WebhookEventType.REMEMBERED)


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_minimal_payload(self) -> None:
        payload = build_payload(
            event=WebhookEventType.REMEMBERED,
            memory_id="mem-1",
            user_id="user1",
        )
        assert payload["event"] == "memory.remembered"
        assert payload["memory_id"] == "mem-1"
        assert payload["user_id"] == "user1"
        assert "timestamp" in payload
        assert "snapshot" not in payload
        assert "previous_state" not in payload

    def test_with_snapshot(self) -> None:
        payload = build_payload(
            event=WebhookEventType.UPDATED,
            memory_id="mem-1",
            user_id="user1",
            snapshot={"content": "hello", "version": 2},
            previous_state={"content": "hi", "version": 1},
        )
        assert payload["snapshot"] == {"content": "hello", "version": 2}
        assert payload["previous_state"] == {"content": "hi", "version": 1}

    def test_with_extra(self) -> None:
        payload = build_payload(
            event=WebhookEventType.CONFLICT,
            memory_id="mem-1",
            user_id="user1",
            extra={"conflict_with": "mem-2"},
        )
        assert payload["conflict_with"] == "mem-2"


# ---------------------------------------------------------------------------
# HMAC signature
# ---------------------------------------------------------------------------


class TestSignPayload:
    def test_signature_is_deterministic(self) -> None:
        payload = {"event": "test", "data": 123}
        sig1 = sign_payload(payload, "secret123")
        sig2 = sign_payload(payload, "secret123")
        assert sig1 == sig2
        assert isinstance(sig1, str)
        assert len(sig1) == 64  # SHA-256 hex

    def test_different_secrets_different_signature(self) -> None:
        payload = {"event": "test"}
        sig1 = sign_payload(payload, "secret-a")
        sig2 = sign_payload(payload, "secret-b")
        assert sig1 != sig2

    def test_empty_secret(self) -> None:
        payload = {"event": "test"}
        sig = sign_payload(payload, "")
        assert isinstance(sig, str)
        assert len(sig) == 64

    def test_signature_verifies(self) -> None:
        import hashlib
        import hmac

        payload = {"event": "test"}
        secret = "my-secret"
        sig = sign_payload(payload, secret)
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected = hmac.new(
            secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        assert sig == expected


# ---------------------------------------------------------------------------
# WebhookStore (CRUD)
# ---------------------------------------------------------------------------


class TestWebhookStore:
    def test_create_and_get(self, store: WebhookStore, sample_config: WebhookConfig) -> None:
        wh_id = store.create(sample_config)
        assert wh_id
        retrieved = store.get(wh_id)
        assert retrieved is not None
        assert retrieved.url == sample_config.url
        assert retrieved.secret == sample_config.secret
        assert retrieved.active is True

    def test_create_auto_assigns_id(self, store: WebhookStore) -> None:
        cfg = WebhookConfig(webhook_id="", url="http://example.com/hook", events=[WebhookEventType.REMEMBERED])  # noqa: E501
        wh_id = store.create(cfg)
        assert wh_id
        assert wh_id != ""

    def test_get_nonexistent(self, store: WebhookStore) -> None:
        assert store.get("nonexistent") is None

    def test_delete(self, store: WebhookStore, sample_config: WebhookConfig) -> None:
        wh_id = store.create(sample_config)
        assert store.delete(wh_id) is True
        assert store.get(wh_id) is None
        # Second delete should return False
        assert store.delete(wh_id) is False

    def test_list_all(self, store: WebhookStore, sample_config: WebhookConfig) -> None:
        store.create(sample_config)
        cfg2 = WebhookConfig(
            webhook_id="",
            url="http://example.com/other",
            events=[WebhookEventType.DELETED],
            active=False,
        )
        store.create(cfg2)
        all_configs = store.list_all(active_only=False)
        assert len(all_configs) == 2

    def test_list_all_active_only(self, store: WebhookStore, sample_config: WebhookConfig) -> None:
        store.create(sample_config)  # active=True
        cfg2 = WebhookConfig(
            webhook_id="",
            url="http://example.com/other",
            events=[WebhookEventType.DELETED],
            active=False,
        )
        store.create(cfg2)
        active_configs = store.list_all(active_only=True)
        assert len(active_configs) == 1
        assert active_configs[0].active is True

    def test_list_for_event(self, store: WebhookStore, sample_config: WebhookConfig) -> None:
        store.create(sample_config)  # REMEMBERED, UPDATED, active
        cfg2 = WebhookConfig(
            webhook_id="",
            url="http://example.com/del",
            events=[WebhookEventType.DELETED],
            active=True,
        )
        store.create(cfg2)
        remembered_hooks = store.list_for_event(WebhookEventType.REMEMBERED)
        assert len(remembered_hooks) == 1
        assert remembered_hooks[0].url == sample_config.url

        deleted_hooks = store.list_for_event(WebhookEventType.DELETED)
        assert len(deleted_hooks) == 1

        conflict_hooks = store.list_for_event(WebhookEventType.CONFLICT)
        assert len(conflict_hooks) == 0

    def test_update(self, store: WebhookStore, sample_config: WebhookConfig) -> None:
        wh_id = store.create(sample_config)
        sample_config.webhook_id = wh_id
        sample_config.url = "http://updated.com/hook"
        sample_config.events = [WebhookEventType.DELETED]
        assert store.update(sample_config) is True
        retrieved = store.get(wh_id)
        assert retrieved is not None
        assert retrieved.url == "http://updated.com/hook"
        assert retrieved.events == [WebhookEventType.DELETED]

    def test_persist_and_reload_events(self, store: WebhookStore) -> None:
        cfg = WebhookConfig(
            webhook_id="",
            url="http://example.com/hook",
            events=[WebhookEventType.REMEMBERED, WebhookEventType.CONFLICT, WebhookEventType.CONSOLIDATED],  # noqa: E501
            secret="s3cr3t",
            active=True,
            retry_config=RetryConfig(max_retries=3, base_delay_seconds=0.5),
        )
        wh_id = store.create(cfg)
        retrieved = store.get(wh_id)
        assert retrieved is not None
        assert retrieved.events == [WebhookEventType.REMEMBERED, WebhookEventType.CONFLICT, WebhookEventType.CONSOLIDATED]  # noqa: E501
        assert retrieved.secret == "s3cr3t"
        assert retrieved.retry_config.max_retries == 3
        assert retrieved.retry_config.base_delay_seconds == 0.5


# ---------------------------------------------------------------------------
# WebhookDispatcher (with mock HTTP server)
# ---------------------------------------------------------------------------


class TestWebhookDispatcher:
    @pytest.fixture
    def store_and_dispatcher(self, store: WebhookStore) -> tuple[WebhookStore, WebhookDispatcher]:
        return store, WebhookDispatcher(store=store)

    @pytest.fixture
    def subscriber(self, store: WebhookStore) -> str:
        """Register a webhook that listens to REMEMBERED events."""
        cfg = WebhookConfig(
            webhook_id="",
            url="http://example.com:1/dummy",  # Will be overridden by patch
            events=[WebhookEventType.REMEMBERED],
            secret="test-secret",
            active=True,
            retry_config=RetryConfig(max_retries=1, base_delay_seconds=0.01),
        )
        return store.create(cfg)

    def test_dispatch_sync_no_subscribers(
        self, store_and_dispatcher: tuple[WebhookStore, WebhookDispatcher]
    ) -> None:
        """dispatch_sync with no subscribers returns empty list."""
        store, dispatcher = store_and_dispatcher
        results = dispatcher.dispatch_sync(
            {"event": "test", "data": 1}, WebhookEventType.CONFLICT
        )
        assert results == []

    def test_dispatch_sync_success(
        self, store: WebhookStore, store_and_dispatcher: tuple[WebhookStore, WebhookDispatcher]
    ) -> None:
        """dispatch_sync with a subscriber calls the endpoint."""
        _, dispatcher = store_and_dispatcher
        # Create a subscriber pointing to an endpoint we control
        cfg = WebhookConfig(
            webhook_id="",
            url="http://example.com:1/test",  # Will fail, so patch httpx
            events=[WebhookEventType.REMEMBERED],
            secret="test",
            active=True,
            retry_config=RetryConfig(max_retries=1, base_delay_seconds=0.01),
        )
        wh_id = store.create(cfg)

        from httpx import Response

        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.post.return_value = Response(200, text="OK")

            results = dispatcher.dispatch_sync(
                build_payload(
                    event=WebhookEventType.REMEMBERED,
                    memory_id="mem-1",
                    user_id="user1",
                ),
                WebhookEventType.REMEMBERED,
            )

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["webhook_id"] == wh_id
        assert results[0]["status_code"] == 200

        # Verify HMAC header
        call_kwargs = mock_client.post.call_args
        assert call_kwargs is not None
        headers = call_kwargs[1]["headers"]
        assert "X-Kemi-Signature" in headers
        assert headers["X-Kemi-Signature"] != ""

    def test_dispatch_sync_signature_matches_body(
        self, store: WebhookStore, store_and_dispatcher: tuple[WebhookStore, WebhookDispatcher]
    ) -> None:
        """The X-Kemi-Signature header must match the HMAC of the exact bytes sent."""
        _, dispatcher = store_and_dispatcher
        cfg = WebhookConfig(
            webhook_id="",
            url="http://example.com:1/sign-test",
            events=[WebhookEventType.REMEMBERED],
            secret="sign-secret",
            active=True,
            retry_config=RetryConfig(max_retries=1, base_delay_seconds=0.01),
        )
        store.create(cfg)

        from httpx import Response

        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.post.return_value = Response(200, text="OK")

            payload = build_payload(
                event=WebhookEventType.REMEMBERED,
                memory_id="mem-1",
                user_id="user1",
                snapshot={"content": "hello"},
            )
            dispatcher.dispatch_sync(payload, WebhookEventType.REMEMBERED)

        call_args = mock_client.post.call_args
        assert call_args is not None
        sent_body = call_args[1]["content"]
        headers = call_args[1]["headers"]
        sent_sig = headers["X-Kemi-Signature"]

        expected_body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        assert sent_body == expected_body

        import hashlib
        import hmac

        expected_sig = hmac.new(
            b"sign-secret", expected_body.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        assert sent_sig == expected_sig

    def test_dispatch_sync_retry_on_failure(
        self, store: WebhookStore, store_and_dispatcher: tuple[WebhookStore, WebhookDispatcher]
    ) -> None:
        """dispatch_sync retries on HTTP error."""
        _, dispatcher = store_and_dispatcher
        cfg = WebhookConfig(
            webhook_id="",
            url="http://example.com:1/fail",
            events=[WebhookEventType.REMEMBERED],
            secret="test",
            active=True,
            retry_config=RetryConfig(max_retries=2, base_delay_seconds=0.01),
        )
        store.create(cfg)

        from httpx import Response

        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.post.return_value = Response(500, text="Server Error")

            results = dispatcher.dispatch_sync(
                build_payload(
                    event=WebhookEventType.REMEMBERED,
                    memory_id="mem-1",
                    user_id="user1",
                ),
                WebhookEventType.REMEMBERED,
            )

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "500" in results[0].get("error", "")
        # Should have retried 2 times
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_dispatch_async_success(
        self, store: WebhookStore, store_and_dispatcher: tuple[WebhookStore, WebhookDispatcher]
    ) -> None:
        """dispatch_async with a subscriber calls the endpoint."""
        _, dispatcher = store_and_dispatcher
        cfg = WebhookConfig(
            webhook_id="",
            url="http://example.com:1/async-test",
            events=[WebhookEventType.UPDATED],
            secret="async-secret",
            active=True,
            retry_config=RetryConfig(max_retries=1, base_delay_seconds=0.01),
        )
        store.create(cfg)

        from httpx import Response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.post.return_value = Response(200, text="OK")

            results = await dispatcher.dispatch_async(
                build_payload(
                    event=WebhookEventType.UPDATED,
                    memory_id="mem-1",
                    user_id="user1",
                ),
                WebhookEventType.UPDATED,
            )

        assert len(results) == 1
        assert results[0]["success"] is True
        # Verify signature
        call_kwargs = mock_client.post.call_args
        assert call_kwargs is not None
        headers = call_kwargs[1]["headers"]
        assert "X-Kemi-Signature" in headers

    @pytest.mark.asyncio
    async def test_dispatch_async_no_subscribers(
        self, store_and_dispatcher: tuple[WebhookStore, WebhookDispatcher]
    ) -> None:
        """dispatch_async with no subscribers returns empty list."""
        _, dispatcher = store_and_dispatcher
        results = await dispatcher.dispatch_async(
            {"event": "test"}, WebhookEventType.CONSOLIDATED
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_dispatch_async_signature_matches_body(
        self, store: WebhookStore, store_and_dispatcher: tuple[WebhookStore, WebhookDispatcher]
    ) -> None:
        """The X-Kemi-Signature header must match the HMAC of the exact bytes sent (async)."""
        _, dispatcher = store_and_dispatcher
        cfg = WebhookConfig(
            webhook_id="",
            url="http://example.com:1/async-sign-test",
            events=[WebhookEventType.UPDATED],
            secret="async-sign-secret",
            active=True,
            retry_config=RetryConfig(max_retries=1, base_delay_seconds=0.01),
        )
        store.create(cfg)

        from httpx import Response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.post.return_value = Response(200, text="OK")

            payload = build_payload(
                event=WebhookEventType.UPDATED,
                memory_id="mem-2",
                user_id="user2",
                snapshot={"content": "world"},
            )
            await dispatcher.dispatch_async(payload, WebhookEventType.UPDATED)

        call_args = mock_client.post.call_args
        assert call_args is not None
        sent_body = call_args[1]["content"]
        headers = call_args[1]["headers"]
        sent_sig = headers["X-Kemi-Signature"]

        expected_body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        assert sent_body == expected_body

        import hashlib
        import hmac

        expected_sig = hmac.new(
            b"async-sign-secret", expected_body.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        assert sent_sig == expected_sig

    @pytest.mark.asyncio
    async def test_dispatch_async_retry(
        self, store: WebhookStore, store_and_dispatcher: tuple[WebhookStore, WebhookDispatcher]
    ) -> None:
        """dispatch_async retries on failure."""
        _, dispatcher = store_and_dispatcher
        cfg = WebhookConfig(
            webhook_id="",
            url="http://example.com:1/retry-test",
            events=[WebhookEventType.DELETED],
            secret="test",
            active=True,
            retry_config=RetryConfig(max_retries=2, base_delay_seconds=0.01),
        )
        store.create(cfg)

        from httpx import Response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.post.return_value = Response(503, text="Unavailable")

            results = await dispatcher.dispatch_async(
                build_payload(
                    event=WebhookEventType.DELETED,
                    memory_id="mem-1",
                    user_id="user1",
                ),
                WebhookEventType.DELETED,
            )

        assert len(results) == 1
        assert results[0]["success"] is False
        assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# Integration: Webhook payload -> storage roundtrip
# ---------------------------------------------------------------------------


class TestWebhookIntegration:
    def test_full_create_list_delete_cycle(self, store: WebhookStore) -> None:
        """Full CRUD cycle for webhooks via the store."""
        cfg = WebhookConfig(
            webhook_id="",
            url="http://example.com/webhook",
            events=[WebhookEventType.REMEMBERED, WebhookEventType.CONFLICT],
            secret="my-secret",
            active=True,
        )
        wh_id = store.create(cfg)
        assert wh_id

        # List
        all_hooks = store.list_all(active_only=False)
        ids = [h.webhook_id for h in all_hooks]
        assert wh_id in ids

        # Get
        retrieved = store.get(wh_id)
        assert retrieved is not None
        assert retrieved.url == "http://example.com/webhook"
        assert len(retrieved.events) == 2
        assert retrieved.active is True

        # Delete
        assert store.delete(wh_id) is True
        assert store.get(wh_id) is None

    def test_multiple_webhooks_same_event(self, store: WebhookStore) -> None:
        """Multiple webhooks can subscribe to the same event."""
        for i in range(3):
            cfg = WebhookConfig(
                webhook_id="",
                url=f"http://example.com/hook{i}",
                events=[WebhookEventType.REMEMBERED],
                secret="test",
                active=True,
            )
            store.create(cfg)

        hooks = store.list_for_event(WebhookEventType.REMEMBERED)
        assert len(hooks) == 3

    def test_payload_with_realistic_data(self) -> None:
        """Build a realistic payload matching what core.py would send."""
        snapshot = {
            "memory_id": "mem-123",
            "content": "Important fact about AI",
            "importance": 0.9,
            "confidence": 1.0,
            "lifecycle_state": "active",
            "memory_type": "semantic",
            "source": "user_stated",
            "tags": ["ai", "knowledge"],
            "namespace": "default",
            "session_id": None,
            "version": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
            "last_accessed_at": "2025-01-01T00:00:00+00:00",
            "metadata": {"source_app": "chat"},
        }
        payload = build_payload(
            event=WebhookEventType.REMEMBERED,
            memory_id="mem-123",
            user_id="user-42",
            snapshot=snapshot,
        )
        assert payload["event"] == "memory.remembered"
        assert payload["snapshot"] == snapshot
        assert payload["user_id"] == "user-42"
        # Verify JSON-serialisable
        json.dumps(payload)
