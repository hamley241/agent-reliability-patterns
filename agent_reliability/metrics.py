"""
Prometheus Metrics for AI Agent Circuit Breakers

Exposes metrics for monitoring circuit breaker state, confidence scores,
token usage, and breaker trip rates.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Placeholder for prometheus_client - not required as dependency
_prometheus_available = False
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _prometheus_available = True
except ImportError:
    pass


class AgentMetrics:
    """
    Prometheus metrics for AI agent reliability monitoring.
    
    Metrics exposed:
    - agent_circuit_state: Current circuit state
    - agent_confidence_score: Histogram of confidence scores
    - agent_tokens_used: Counter of tokens consumed
    - agent_breaker_trips_total: Counter of circuit breaker trips
    """
    
    def __init__(self, prefix: str = "agent"):
        self.prefix = prefix
        self._initialized = False
        
        if not _prometheus_available:
            logger.warning(
                "prometheus_client not installed. "
                "Install with: pip install prometheus-client"
            )
            return
        
        # Circuit state gauge
        self.circuit_state = Gauge(
            f"{prefix}_circuit_state",
            "Current circuit breaker state (0=closed, 1=reasoning_open, 2=context_open, 3=half_open)",
            ["agent_id"]
        )
        
        # Confidence score histogram
        self.confidence_score = Histogram(
            f"{prefix}_confidence_score",
            "Distribution of agent confidence scores",
            ["agent_id"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        # Token usage counter
        self.tokens_used = Counter(
            f"{prefix}_tokens_used_total",
            "Total tokens consumed by agent",
            ["agent_id"]
        )
        
        # Breaker trip counter
        self.breaker_trips = Counter(
            f"{prefix}_breaker_trips_total",
            "Total circuit breaker trips",
            ["agent_id", "reason"]
        )
        
        self._initialized = True
    
    def record_confidence(self, agent_id: str, confidence: float):
        """Record a confidence score."""
        if self._initialized:
            self.confidence_score.labels(agent_id=agent_id).observe(confidence)
    
    def record_tokens(self, agent_id: str, tokens: int):
        """Record token usage."""
        if self._initialized:
            self.tokens_used.labels(agent_id=agent_id).inc(tokens)
    
    def record_trip(self, agent_id: str, reason: str):
        """Record a circuit breaker trip."""
        if self._initialized:
            self.breaker_trips.labels(agent_id=agent_id, reason=reason).inc()
    
    def set_state(self, agent_id: str, state: int):
        """Set current circuit state (0=closed, 1=reasoning_open, 2=context_open, 3=half_open)."""
        if self._initialized:
            self.circuit_state.labels(agent_id=agent_id).set(state)


# Global metrics instance
_metrics: Optional[AgentMetrics] = None


def setup_metrics(port: int = 9090, prefix: str = "agent") -> AgentMetrics:
    """
    Initialize and start Prometheus metrics server.
    
    Args:
        port: Port to expose metrics on
        prefix: Metric name prefix
        
    Returns:
        AgentMetrics instance
    """
    global _metrics
    
    if not _prometheus_available:
        logger.error(
            "Cannot setup metrics: prometheus_client not installed. "
            "Install with: pip install prometheus-client"
        )
        return AgentMetrics(prefix)
    
    _metrics = AgentMetrics(prefix)
    
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
    
    return _metrics


def get_metrics() -> Optional[AgentMetrics]:
    """Get the global metrics instance."""
    return _metrics
