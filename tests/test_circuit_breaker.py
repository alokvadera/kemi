"""Tests for circuit breaker pattern in OpenAI embedding adapter."""

from unittest.mock import patch

import pytest

# Import the circuit breaker and adapter
from kemi.adapters.embedding.openai import (
    CircuitBreaker,
    CircuitState,
    OpenAIEmbedAdapter,
)


class TestCircuitBreaker:
    """Unit tests for CircuitBreaker class."""

    def setup_method(self):
        """Reset circuit breaker state before each test."""
        self.cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

    def test_initial_state_is_closed(self):
        """Test that circuit starts in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_allow_request_in_closed_state(self):
        """Test that requests are allowed in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        assert cb.allow_request() is True

    def test_success_resets_failure_count(self):
        """Test that recording success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 2

        cb.record_success()
        assert cb._failure_count == 0

    def test_failure_increments_count(self):
        """Test that recording failure increments count."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=10.0)

        for i in range(3):
            cb.record_failure()
            assert cb._failure_count == i + 1

    def test_failure_threshold_opens_circuit(self):
        """Test that reaching failure threshold opens circuit."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()  # Now at threshold
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_recovery_timeout_transitions_to_half_open(self):
        """Test that after recovery timeout, circuit goes to HALF_OPEN."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        import time

        time.sleep(0.15)

        # Should transition to HALF_OPEN on next state access
        state = cb.state
        assert state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        """Test that success in HALF_OPEN closes the circuit."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,
            half_open_max_requests=1,
        )

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout and access state to transition
        import time

        time.sleep(0.15)
        state = cb.state
        assert state == CircuitState.HALF_OPEN

        # Record success in half-open
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        """Test that failure in HALF_OPEN reopens the circuit."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,
            half_open_max_requests=1,
        )

        # Open and transition to half-open
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()

        import time

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN  # Trigger transition

        # Record failure in half-open
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_max_requests_limits(self):
        """Test that half_open_max_requests limits concurrent test requests."""
        import time

        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,
            half_open_max_requests=1,
        )

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout to transition to half-open
        time.sleep(0.15)

        # First request allowed (transitions to half-open)
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

        # Second request blocked (half_open_max_requests=1 exhausted)
        assert cb.allow_request() is False

    def test_get_state_returns_dict(self):
        """Test that get_state returns proper state info."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        state_info = cb.get_state()
        assert isinstance(state_info, dict)
        assert "state" in state_info
        assert "failure_count" in state_info
        assert "failure_threshold" in state_info
        assert "recovery_timeout" in state_info
        assert state_info["state"] == "closed"
        assert state_info["failure_count"] == 0

    def test_get_state_shows_failure_count_at_threshold(self):
        """Test that get_state shows failure count when circuit is open."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        state_info = cb.get_state()
        assert state_info["state"] == "open"
        assert state_info["failure_count"] == 3

    def test_thread_safety(self):
        """Test that circuit breaker operations are thread-safe."""
        import threading

        cb = CircuitBreaker(failure_threshold=100, recovery_timeout=10.0)

        def record_failures():
            for _ in range(50):
                cb.record_failure()

        threads = [threading.Thread(target=record_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded 200 failures and opened the circuit
        assert cb.state == CircuitState.OPEN
        assert cb._failure_count >= 100


class TestCircuitBreakerEdgeCases:
    """Edge case tests for CircuitBreaker."""

    def test_record_success_when_already_closed_does_nothing(self):
        """Test that recording success when already closed is safe."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        # Should not raise
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_allow_request_when_half_open_exhausted(self):
        """Test allow_request when half_open_requests exhausted."""
        import time

        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,
            half_open_max_requests=1,
        )

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()

        # Wait for recovery timeout to transition to half-open
        time.sleep(0.15)

        # First request transitions to half-open
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

        # Second request should be blocked
        assert cb.allow_request() is False

    def test_multiple_successive_failures_in_closed_state(self):
        """Test multiple successive failures in closed state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Additional failures after open should not change state
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestOpenAIEmbedAdapterCircuitBreaker:
    """Tests for circuit breaker integration in OpenAIEmbedAdapter."""

    def test_adapter_has_circuit_breaker(self):
        """Test that OpenAIEmbedAdapter has circuit breaker."""
        adapter = OpenAIEmbedAdapter(api_key="test-key")
        assert hasattr(adapter, "_circuit_breaker")
        assert isinstance(adapter._circuit_breaker, CircuitBreaker)

    def test_get_circuit_breaker_state(self):
        """Test get_circuit_breaker_state method."""
        adapter = OpenAIEmbedAdapter(api_key="test-key")
        state = adapter.get_circuit_breaker_state()

        assert isinstance(state, dict)
        assert "state" in state
        assert "failure_count" in state

    def test_adapter_circuit_breaker_configurable(self):
        """Test that circuit breaker params are configurable."""
        adapter = OpenAIEmbedAdapter(
            api_key="test-key",
            cb_failure_threshold=10,
            cb_recovery_timeout=60.0,
        )

        cb_state = adapter.get_circuit_breaker_state()
        assert cb_state["failure_threshold"] == 10
        assert cb_state["recovery_timeout"] == 60.0

    def test_embed_raises_when_circuit_open(self):
        """Test that embed raises RuntimeError when circuit is open."""
        adapter = OpenAIEmbedAdapter(
            api_key="test-key",
            cb_failure_threshold=2,
        )

        # Force circuit open
        adapter._circuit_breaker.record_failure()
        adapter._circuit_breaker.record_failure()
        assert adapter._circuit_breaker.state == CircuitState.OPEN

        # embed should raise RuntimeError
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            adapter.embed(["test text"])

    def test_embed_success_records_success(self):
        """Test that successful embed records success."""
        adapter = OpenAIEmbedAdapter(api_key="test-key")

        # Mock _TENACITY_AVAILABLE to False to force simple fallback path
        with patch("kemi.adapters.embedding.openai._TENACITY_AVAILABLE", False):
            with patch.object(adapter, "_make_embedding_request", return_value=[[0.1] * 1536]):
                adapter.embed(["test text"])

        # Circuit breaker should have recorded success
        state = adapter.get_circuit_breaker_state()
        assert state["failure_count"] == 0  # Reset on success

    def test_embed_failure_records_failure(self):
        """Test that failed embed records failure."""
        adapter = OpenAIEmbedAdapter(
            api_key="test-key",
            cb_failure_threshold=5,
        )

        # Mock _TENACITY_AVAILABLE to False to force fallback path
        with patch("kemi.adapters.embedding.openai._TENACITY_AVAILABLE", False):
            # Mock to raise a retryable error
            error = Exception("rate limit exceeded")
            with patch.object(adapter, "_make_embedding_request", side_effect=error):
                with patch.object(adapter, "_is_openai_retryable", return_value=True):
                    # Use fallback path so we can test failure recording
                    try:
                        adapter.embed(["test text"])
                    except Exception:
                        pass

        # Should have recorded failure
        state = adapter.get_circuit_breaker_state()
        assert state["failure_count"] >= 1

    def test_circuit_breaker_state_in_error_message(self):
        """Test that circuit state info is included in error message."""
        adapter = OpenAIEmbedAdapter(
            api_key="test-key",
            cb_failure_threshold=2,
            cb_recovery_timeout=30.0,
        )

        # Open the circuit
        adapter._circuit_breaker.record_failure()
        adapter._circuit_breaker.record_failure()

        with pytest.raises(RuntimeError) as exc_info:
            adapter.embed(["test text"])

        error_msg = str(exc_info.value)
        assert "open" in error_msg.lower()
        assert "30" in error_msg  # recovery_timeout
