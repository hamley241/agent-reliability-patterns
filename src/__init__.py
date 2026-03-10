"""Agent Reliability Patterns - Production reliability for AI agents."""

from .circuit_breaker import (
    AIAgentCircuitBreaker,
    CircuitState,
    AgentResponse,
    Config,
)
from .confidence import (
    SecondaryModelEvaluator,
    SelfConsistencyChecker,
    LogitBasedHeuristics,
)
from .fallbacks import (
    ClarificationFallback,
    ContextResetFallback,
    EscalationFallback,
    FallbackResponse,
)

__version__ = "0.1.0"
__all__ = [
    "AIAgentCircuitBreaker",
    "CircuitState",
    "AgentResponse",
    "Config",
    "SecondaryModelEvaluator",
    "SelfConsistencyChecker",
    "LogitBasedHeuristics",
    "ClarificationFallback",
    "ContextResetFallback",
    "EscalationFallback",
    "FallbackResponse",
]
