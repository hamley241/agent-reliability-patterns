"""Tests for AI Agent Circuit Breaker."""

import pytest
import time
from src.circuit_breaker import (
    AIAgentCircuitBreaker,
    CircuitState,
    AgentResponse,
    Config,
)


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""
    
    def test_valid_response(self):
        response = AgentResponse(
            text="Hello, world!",
            confidence=0.85,
            token_count=10
        )
        assert response.confidence == 0.85
        assert response.token_count == 10
    
    def test_invalid_confidence_high(self):
        with pytest.raises(ValueError):
            AgentResponse(text="test", confidence=1.5, token_count=10)
    
    def test_invalid_confidence_low(self):
        with pytest.raises(ValueError):
            AgentResponse(text="test", confidence=-0.1, token_count=10)
    
    def test_invalid_token_count(self):
        with pytest.raises(ValueError):
            AgentResponse(text="test", confidence=0.5, token_count=-1)


class TestCircuitBreakerInitialization:
    """Tests for circuit breaker initialization."""
    
    def test_default_initialization(self):
        breaker = AIAgentCircuitBreaker()
        assert breaker.state == CircuitState.REASONING_CLOSED
        assert breaker.config.confidence_threshold == 0.5
        assert breaker.config.token_limit_percent == 0.8
    
    def test_custom_initialization(self):
        breaker = AIAgentCircuitBreaker(
            confidence_threshold=0.6,
            token_limit_percent=0.7,
            recovery_timeout=60,
        )
        assert breaker.config.confidence_threshold == 0.6
        assert breaker.config.token_limit_percent == 0.7
        assert breaker.config.recovery_timeout_seconds == 60
    
    def test_config_object(self):
        config = Config(
            confidence_threshold=0.4,
            confidence_window_size=10,
        )
        breaker = AIAgentCircuitBreaker(config=config)
        assert breaker.config.confidence_threshold == 0.4
        assert breaker.config.confidence_window_size == 10


class TestConfidenceDegradation:
    """Tests for confidence-based circuit tripping."""
    
    def test_no_trip_high_confidence(self):
        breaker = AIAgentCircuitBreaker(confidence_threshold=0.5)
        
        for _ in range(5):
            response = AgentResponse(text="test", confidence=0.8, token_count=10)
            assert breaker.should_trip(response) is False
        
        assert breaker.state == CircuitState.REASONING_CLOSED
    
    def test_trip_on_low_confidence(self):
        breaker = AIAgentCircuitBreaker(confidence_threshold=0.5)
        
        # Need at least 3 responses for trend detection
        responses = [
            AgentResponse(text="test", confidence=0.4, token_count=10),
            AgentResponse(text="test", confidence=0.3, token_count=10),
            AgentResponse(text="test", confidence=0.2, token_count=10),
        ]
        
        for i, response in enumerate(responses):
            result = breaker.should_trip(response)
            if i == 2:  # Should trip on third low-confidence response
                assert result is True
        
        assert breaker.state == CircuitState.REASONING_OPEN
    
    def test_gradual_degradation(self):
        breaker = AIAgentCircuitBreaker(confidence_threshold=0.5)
        
        # Start high, gradually decrease
        confidences = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2]
        
        for conf in confidences:
            response = AgentResponse(text="test", confidence=conf, token_count=10)
            breaker.should_trip(response)
        
        # Should have tripped at some point
        assert breaker.state == CircuitState.REASONING_OPEN


class TestContextExhaustion:
    """Tests for token/context-based circuit tripping."""
    
    def test_trip_on_token_exhaustion(self):
        breaker = AIAgentCircuitBreaker(
            token_limit_percent=0.8,
            max_context_tokens=1000,
        )
        
        # Use 850 tokens (85% > 80% limit)
        response = AgentResponse(text="test", confidence=0.9, token_count=850)
        result = breaker.should_trip(response)
        
        assert result is True
        assert breaker.state == CircuitState.CONTEXT_OPEN
    
    def test_no_trip_under_limit(self):
        breaker = AIAgentCircuitBreaker(
            token_limit_percent=0.8,
            max_context_tokens=1000,
        )
        
        response = AgentResponse(text="test", confidence=0.9, token_count=500)
        result = breaker.should_trip(response)
        
        assert result is False
        assert breaker.state == CircuitState.REASONING_CLOSED
    
    def test_cumulative_token_tracking(self):
        breaker = AIAgentCircuitBreaker(
            token_limit_percent=0.8,
            max_context_tokens=1000,
        )
        
        # Multiple small responses that add up
        for i in range(9):
            response = AgentResponse(text="test", confidence=0.9, token_count=100)
            result = breaker.should_trip(response)
            
            if i < 7:  # Under 800 tokens
                assert result is False
            else:  # 800+ tokens
                assert result is True
                break


class TestCircularReasoningDetection:
    """Tests for circular reasoning detection."""
    
    def test_detect_contradiction(self):
        breaker = AIAgentCircuitBreaker()
        
        response = AgentResponse(
            text="Actually, I was wrong about that earlier.",
            confidence=0.7,
            token_count=50,
        )
        result = breaker.should_trip(response)
        
        assert result is True
        assert breaker.state == CircuitState.REASONING_OPEN
    
    def test_detect_reconsideration(self):
        breaker = AIAgentCircuitBreaker()
        
        response = AgentResponse(
            text="Let me reconsider this approach.",
            confidence=0.8,
            token_count=50,
        )
        result = breaker.should_trip(response)
        
        assert result is True
    
    def test_no_false_positive(self):
        breaker = AIAgentCircuitBreaker()
        
        response = AgentResponse(
            text="The answer to your question is 42.",
            confidence=0.9,
            token_count=50,
        )
        result = breaker.should_trip(response)
        
        assert result is False


class TestRecovery:
    """Tests for circuit recovery."""
    
    def test_recovery_attempt_timing(self):
        breaker = AIAgentCircuitBreaker(recovery_timeout=1)  # 1 second
        
        # Trip the breaker
        response = AgentResponse(
            text="Actually, I was wrong",
            confidence=0.5,
            token_count=10,
        )
        breaker.should_trip(response)
        assert breaker.state == CircuitState.REASONING_OPEN
        
        # Too early for recovery
        assert breaker.attempt_recovery() is False
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Now should allow recovery
        assert breaker.attempt_recovery() is True
        assert breaker.state == CircuitState.HALF_OPEN
    
    def test_successful_recovery(self):
        breaker = AIAgentCircuitBreaker(recovery_timeout=0)  # Immediate
        
        # Trip the breaker
        breaker._trip_breaker(CircuitState.REASONING_OPEN, "test")
        
        # Attempt recovery
        breaker.attempt_recovery()
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Successful request closes the circuit
        response = AgentResponse(text="Good response", confidence=0.9, token_count=10)
        breaker.should_trip(response)
        
        assert breaker.state == CircuitState.REASONING_CLOSED
    
    def test_failed_recovery(self):
        breaker = AIAgentCircuitBreaker(recovery_timeout=0)
        
        # Trip and attempt recovery
        breaker._trip_breaker(CircuitState.REASONING_OPEN, "test")
        breaker.attempt_recovery()
        
        # Failed request during HALF_OPEN
        response = AgentResponse(
            text="Wait, that's not right",
            confidence=0.3,
            token_count=10,
        )
        breaker.should_trip(response)
        
        # Should be back to REASONING_OPEN
        assert breaker.state == CircuitState.REASONING_OPEN


class TestCallbacks:
    """Tests for callback functionality."""
    
    def test_state_change_callback(self):
        breaker = AIAgentCircuitBreaker()
        
        state_changes = []
        
        def on_change(old, new):
            state_changes.append((old, new))
        
        breaker.on_state_change(on_change)
        
        # Trip the breaker
        response = AgentResponse(
            text="Actually, I was wrong",
            confidence=0.5,
            token_count=10,
        )
        breaker.should_trip(response)
        
        assert len(state_changes) == 1
        assert state_changes[0] == (CircuitState.REASONING_CLOSED, CircuitState.REASONING_OPEN)
    
    def test_trip_callback(self):
        breaker = AIAgentCircuitBreaker()
        
        trips = []
        
        def on_trip(state, reason):
            trips.append((state, reason))
        
        breaker.on_trip(on_trip)
        
        # Trip via circular reasoning
        response = AgentResponse(
            text="Let me reconsider",
            confidence=0.8,
            token_count=10,
        )
        breaker.should_trip(response)
        
        assert len(trips) == 1
        assert trips[0][0] == CircuitState.REASONING_OPEN
        assert trips[0][1] == "circular_reasoning"


class TestStatistics:
    """Tests for statistics tracking."""
    
    def test_stats_tracking(self):
        breaker = AIAgentCircuitBreaker()
        
        # Some successful requests
        for _ in range(5):
            response = AgentResponse(text="Good", confidence=0.9, token_count=100)
            breaker.should_trip(response)
        
        # One failed request
        response = AgentResponse(
            text="Actually, I was wrong",
            confidence=0.5,
            token_count=50,
        )
        breaker.should_trip(response)
        
        stats = breaker.get_stats()
        
        assert stats["successful_requests"] == 5
        assert stats["failed_requests"] == 1
        assert stats["trip_count"] == 1
        assert stats["token_usage"] == 550
    
    def test_reset(self):
        breaker = AIAgentCircuitBreaker()
        
        # Use the breaker
        response = AgentResponse(text="test", confidence=0.5, token_count=100)
        breaker.should_trip(response)
        
        # Reset
        breaker.reset()
        
        assert breaker.state == CircuitState.REASONING_CLOSED
        assert breaker.token_usage == 0
        assert len(breaker.confidence_window) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
