"""
AI Agent Circuit Breaker

Detects reasoning failures, not just network failures.
Inspired by Netflix Hystrix and AWS Lambda health patterns.
"""

from enum import Enum
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import time
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states for AI agents."""
    REASONING_CLOSED = "reasoning_closed"   # Normal operation
    REASONING_OPEN = "reasoning_open"       # Confidence degradation detected
    CONTEXT_OPEN = "context_open"           # Token/context limit reached
    HALF_OPEN = "half_open"                 # Testing recovery


@dataclass
class AgentResponse:
    """Represents an AI agent's response with quality metrics."""
    text: str
    confidence: float  # 0.0 to 1.0
    token_count: int
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.token_count < 0:
            raise ValueError(f"Token count must be non-negative, got {self.token_count}")


@dataclass
class Config:
    """Configuration for the circuit breaker."""
    confidence_threshold: float = 0.5
    confidence_window_size: int = 5
    token_limit_percent: float = 0.8
    recovery_timeout_seconds: int = 30
    max_context_tokens: int = 4096
    
    # Circular reasoning detection
    contradiction_signals: List[str] = field(default_factory=lambda: [
        "actually, i was wrong",
        "let me reconsider",
        "on second thought",
        "wait, that's not right",
        "i need to correct myself",
    ])


class AIAgentCircuitBreaker:
    """
    Circuit breaker for AI agents - detects reasoning failures.
    
    Unlike traditional circuit breakers that detect network failures,
    this detects reasoning quality degradation:
    - Confidence drops over sliding window
    - Token/context exhaustion
    - Circular reasoning patterns
    
    Example:
        >>> breaker = AIAgentCircuitBreaker(confidence_threshold=0.5)
        >>> response = AgentResponse(text="...", confidence=0.3, token_count=100)
        >>> if breaker.should_trip(response):
        ...     return request_clarification()
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        token_limit_percent: float = 0.8,
        recovery_timeout: int = 30,
        max_context_tokens: int = 4096,
        config: Optional[Config] = None,
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            confidence_threshold: Trip when avg confidence falls below this
            token_limit_percent: Trip when token usage exceeds this percentage
            recovery_timeout: Seconds to wait before attempting recovery
            max_context_tokens: Maximum context window size
            config: Optional full configuration object
        """
        if config:
            self.config = config
        else:
            self.config = Config(
                confidence_threshold=confidence_threshold,
                token_limit_percent=token_limit_percent,
                recovery_timeout_seconds=recovery_timeout,
                max_context_tokens=max_context_tokens,
            )
        
        self.state = CircuitState.REASONING_CLOSED
        self.confidence_window = deque(maxlen=self.config.confidence_window_size)
        self.token_usage = 0
        self.last_trip_time: Optional[float] = None
        self.trip_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Callbacks
        self._on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
        self._on_trip: Optional[Callable[[CircuitState, str], None]] = None
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is in any open state."""
        return self.state in (CircuitState.REASONING_OPEN, CircuitState.CONTEXT_OPEN)
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is testing recovery."""
        return self.state == CircuitState.HALF_OPEN
    
    def should_trip(self, response: AgentResponse) -> bool:
        """
        Check if circuit should trip based on response quality.
        
        Args:
            response: The agent's response with quality metrics
            
        Returns:
            True if the circuit should trip (stop reasoning)
        """
        # Update metrics
        self.confidence_window.append(response.confidence)
        self.token_usage += response.token_count
        
        # Check for confidence degradation
        if self._confidence_trending_down():
            self._trip_breaker(CircuitState.REASONING_OPEN, "confidence_degradation")
            return True
        
        # Check for token/context exhaustion
        if self._context_exhausted():
            self._trip_breaker(CircuitState.CONTEXT_OPEN, "context_exhaustion")
            return True
        
        # Check for circular reasoning patterns
        if self._detects_circular_reasoning(response):
            self._trip_breaker(CircuitState.REASONING_OPEN, "circular_reasoning")
            return True
        
        # Request succeeded
        self.successful_requests += 1
        
        # If in HALF_OPEN and succeeded, close the circuit
        if self.state == CircuitState.HALF_OPEN:
            self._close_circuit()
        
        return False
    
    def _confidence_trending_down(self) -> bool:
        """Detect confidence degradation using sliding window."""
        if len(self.confidence_window) < 3:
            return False
        
        # Calculate recent average
        recent = list(self.confidence_window)[-3:]
        recent_avg = sum(recent) / len(recent)
        
        # Trip if below threshold
        if recent_avg < self.config.confidence_threshold:
            logger.info(
                f"Confidence trending down: avg={recent_avg:.2f}, "
                f"threshold={self.config.confidence_threshold}"
            )
            return True
        
        return False
    
    def _context_exhausted(self) -> bool:
        """Check if token usage exceeds limit."""
        usage_percent = self.token_usage / self.config.max_context_tokens
        if usage_percent > self.config.token_limit_percent:
            logger.info(
                f"Context exhausted: usage={usage_percent:.1%}, "
                f"limit={self.config.token_limit_percent:.1%}"
            )
            return True
        return False
    
    def _detects_circular_reasoning(self, response: AgentResponse) -> bool:
        """
        Detect contradiction and circular reasoning patterns.
        
        Note: This uses simple heuristics. Production implementations
        should use semantic similarity or embedding-based detection.
        """
        text_lower = response.text.lower()
        for signal in self.config.contradiction_signals:
            if signal in text_lower:
                logger.info(f"Circular reasoning detected: found '{signal}'")
                return True
        return False
    
    def _trip_breaker(self, new_state: CircuitState, reason: str):
        """Trip the circuit breaker to an open state."""
        old_state = self.state
        self.state = new_state
        self.last_trip_time = time.time()
        self.trip_count += 1
        self.failed_requests += 1
        
        logger.warning(f"Circuit tripped: {old_state.value} -> {new_state.value} ({reason})")
        
        if self._on_state_change:
            self._on_state_change(old_state, new_state)
        if self._on_trip:
            self._on_trip(new_state, reason)
    
    def _close_circuit(self):
        """Close the circuit after successful recovery."""
        old_state = self.state
        self.state = CircuitState.REASONING_CLOSED
        self.confidence_window.clear()
        
        logger.info(f"Circuit closed: {old_state.value} -> {self.state.value}")
        
        if self._on_state_change:
            self._on_state_change(old_state, self.state)
    
    def attempt_recovery(self) -> bool:
        """
        Check if we should attempt recovery from open state.
        
        Returns:
            True if recovery attempt should proceed
        """
        if self.state not in (CircuitState.REASONING_OPEN, CircuitState.CONTEXT_OPEN):
            return False
        
        if self.last_trip_time is None:
            return False
        
        elapsed = time.time() - self.last_trip_time
        if elapsed >= self.config.recovery_timeout_seconds:
            old_state = self.state
            self.state = CircuitState.HALF_OPEN
            logger.info(f"Attempting recovery: {old_state.value} -> half_open")
            
            if self._on_state_change:
                self._on_state_change(old_state, self.state)
            
            return True
        
        return False
    
    def reset(self):
        """Reset the circuit breaker to initial state."""
        self.state = CircuitState.REASONING_CLOSED
        self.confidence_window.clear()
        self.token_usage = 0
        self.last_trip_time = None
        logger.info("Circuit breaker reset")
    
    def reset_token_usage(self):
        """Reset token usage counter (e.g., after context compression)."""
        self.token_usage = 0
    
    def on_state_change(self, callback: Callable[[CircuitState, CircuitState], None]):
        """Register callback for state changes."""
        self._on_state_change = callback
    
    def on_trip(self, callback: Callable[[CircuitState, str], None]):
        """Register callback for circuit trips."""
        self._on_trip = callback
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "trip_count": self.trip_count,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "token_usage": self.token_usage,
            "token_limit": self.config.max_context_tokens,
            "confidence_window": list(self.confidence_window),
        }
