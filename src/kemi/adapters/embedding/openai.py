import logging
import os
import time
from enum import Enum
from threading import Lock
from typing import Any

from kemi.adapters.base import EmbeddingAdapter
from kemi.exceptions import ConfigurationError, EmbeddingError

# Tenacity for robust retry logic with exponential backoff
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential_jitter,
    )

    _TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TENACITY_AVAILABLE = False


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failing fast, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery, limited requests


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when a service is down.

    States:
    - CLOSED: Normal operation. Failures are counted.
    - OPEN: Too many failures. Requests fail immediately without trying.
    - HALF_OPEN: After recovery_timeout, allow one test request.

    Args:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Seconds to wait before trying recovery.
        half_open_max_requests: Max requests in half-open state.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_requests: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_requests = half_open_max_requests

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_requests = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if self._last_failure_time is not None:
                    if time.time() - self._last_failure_time >= self._recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_requests = 0
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request is allowed, False if circuit is open.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time is not None:
                    if time.time() - self._last_failure_time >= self._recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_requests = 1  # Count this probe request
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests < self._half_open_max_requests:
                    self._half_open_requests += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        """Record a successful request.

        In CLOSED state: reset failure count.
        In HALF_OPEN state: transition to CLOSED.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                self._failure_count = 0

            elif self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._half_open_max_requests:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0

    def record_failure(self) -> None:
        """Record a failed request.

        In CLOSED state: increment failure count, open circuit if threshold reached.
        In HALF_OPEN state: immediately open circuit.
        """
        with self._lock:
            self._last_failure_time = time.time()

            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    logger = logging.getLogger(__name__)
                    logger.error(f"Circuit breaker opened after {self._failure_count} failures")

            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0

    def get_state(self) -> dict[str, Any]:
        """Get circuit breaker state info for monitoring."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "failure_threshold": self._failure_threshold,
                "recovery_timeout": self._recovery_timeout,
            }


class OpenAIEmbedAdapter(EmbeddingAdapter):
    """Embedding adapter using OpenAI's embedding API (or compatible providers).

    Default model: text-embedding-3-small
    Default dimension: 1536

    Features:
    - Automatic retry with exponential backoff for transient errors
    - Configurable timeout for requests
    - Rate limit handling (429 errors)
    - OpenAI-compatible provider support via base_url (e.g., Tokenlb, proxies)

    Args:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        base_url: Base URL for OpenAI-compatible endpoints. Defaults to
            OPENAI_BASE_URL env var. Use this to connect to providers like
            Tokenlb, local proxies, or other OpenAI-compatible services.
        model_name: Model to use. Defaults to text-embedding-3-small.
        timeout: Request timeout in seconds. Defaults to 60.
        max_retries: Maximum retry attempts for transient errors. Defaults to 3.
        initial_delay: Initial backoff delay in seconds. Defaults to 1.0.
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSION = 1536
    DEFAULT_TIMEOUT = 60.0  # seconds
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_INITIAL_DELAY = 1.0  # seconds

    # Circuit breaker default settings
    DEFAULT_CB_FAILURE_THRESHOLD = 5
    DEFAULT_CB_RECOVERY_TIMEOUT = 30.0  # seconds
    DEFAULT_CB_HALF_OPEN_MAX = 1

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        initial_delay: float | None = None,
        cb_failure_threshold: int | None = None,
        cb_recovery_timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._model_name = model_name or self.DEFAULT_MODEL
        self._timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self._max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        self._initial_delay = (
            initial_delay if initial_delay is not None else self.DEFAULT_INITIAL_DELAY
        )
        self._client: Any = None

        # Circuit breaker configuration
        cb_threshold = (
            cb_failure_threshold
            if cb_failure_threshold is not None
            else self.DEFAULT_CB_FAILURE_THRESHOLD
        )
        cb_timeout = (
            cb_recovery_timeout
            if cb_recovery_timeout is not None
            else self.DEFAULT_CB_RECOVERY_TIMEOUT
        )
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=cb_threshold,
            recovery_timeout=cb_timeout,
            half_open_max_requests=self.DEFAULT_CB_HALF_OPEN_MAX,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI

                kwargs: dict[str, Any] = {
                    "api_key": self._api_key,
                    "timeout": self._timeout,
                    "max_retries": 0,  # We handle retries ourselves via Tenacity
                }
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = OpenAI(**kwargs)
            except ImportError as e:
                raise ConfigurationError("openai not installed. Run: pip install kemi[openai]") from e  # noqa: E501
        return self._client

    def _is_openai_retryable(self, error: Exception) -> bool:
        """Determine if an error is transient and worth retrying."""
        error_str = str(error).lower()

        # Rate limit errors
        if "429" in error_str or "rate limit" in error_str:
            return True

        # Server errors
        if any(code in error_str for code in ["500", "502", "503", "504", "server error"]):
            return True

        # Timeout errors
        if "timeout" in error_str or "timed out" in error_str:
            return True

        # Connection errors
        if "connection" in error_str or "network" in error_str:
            return True

        # Check for specific OpenAI error types if available
        if hasattr(error, "status_code"):
            status_code = getattr(error, "status_code", None)
            if status_code in (429, 500, 502, 503, 504):
                return True

        return False

    def _make_embedding_request(self, texts: list[str]) -> list[list[float]]:
        """Make a single embedding request. Used by Tenacity retry."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with automatic retry on transient errors.

        Retries on:
        - Rate limit errors (429)
        - Server errors (500, 502, 503, 504)
        - Timeout errors
        - Connection errors

        Uses Tenacity for robust exponential backoff with jitter.
        Circuit breaker prevents cascading failures when service is down.
        """
        if not texts:
            return []

        # Check circuit breaker state
        if not self._circuit_breaker.allow_request():
            state_info = self._circuit_breaker.get_state()
            raise EmbeddingError(
                f"Circuit breaker is {state_info['state']}. "
                f"Failure count: {state_info['failure_count']}, "
                f"Recovery timeout: {state_info['recovery_timeout']}s"
            )

        # Check if Tenacity is available for enhanced retry logic
        if _TENACITY_AVAILABLE:
            # Use Tenacity decorator for declarative retry logic
            # Custom retry condition only retries on retryable errors
            retry_decorator = retry(
                stop=stop_after_attempt(self._max_retries + 1),
                wait=wait_exponential_jitter(
                    initial=self._initial_delay,
                    max=30.0,  # Cap maximum wait at 30 seconds
                ),
                retry=self._tenacity_retry_condition,
                reraise=True,
                before_sleep=lambda retry_state: self._log_retry(retry_state),
            )

            try:
                return retry_decorator(self._make_embedding_request)(texts)
            except Exception:
                self._circuit_breaker.record_failure()
                raise

        # Fallback to simple retry without Tenacity
        last_exception: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                result = self._make_embedding_request(texts)
                self._circuit_breaker.record_success()
                return result
            except Exception as e:
                last_exception = e

                if not self._is_openai_retryable(e):
                    self._circuit_breaker.record_failure()
                    raise

                if attempt < self._max_retries:
                    delay = self._initial_delay * (2**attempt)
                    time.sleep(delay)

        if last_exception is not None:
            self._circuit_breaker.record_failure()
            raise last_exception
        raise EmbeddingError("OpenAI embedding failed after retries")

    def _on_success(self) -> None:
        """Called after successful embedding request to record success."""
        self._circuit_breaker.record_success()

    def _tenacity_retry_condition(self, retry_state: Any) -> bool:
        """Custom Tenacity retry condition - only retry if error is retryable."""
        if retry_state.outcome is None:
            return False
        exception = retry_state.outcome.exception()
        if exception is None:
            return False
        return self._is_openai_retryable(exception)

    def _tenacity_after(self, retry_state: Any) -> None:
        """Called after Tenacity retry attempt - record success on success."""
        if retry_state.outcome and retry_state.outcome.exception() is None:
            self._circuit_breaker.record_success()

    def _log_retry(self, retry_state: Any) -> None:
        """Log retry attempts for monitoring."""
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Retrying OpenAI embedding: attempt {retry_state.attempt_number}, "
            f"wait={retry_state.next_action.sleep if retry_state.next_action else 'N/A'}s"
        )

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]

    def get_circuit_breaker_state(self) -> dict[str, Any]:
        """Get circuit breaker state for monitoring/debugging."""
        return self._circuit_breaker.get_state()

    def dimension(self) -> int:
        """Return the embedding dimension for this model."""
        return self.DEFAULT_DIMENSION
