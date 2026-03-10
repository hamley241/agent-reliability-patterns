"""
Fallback Strategies for AI Agent Circuit Breakers

When the circuit trips, these determine what to do instead of continuing
to reason poorly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class FallbackResponse:
    """Response from a fallback strategy."""
    action: str
    message: str
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for fallback decisions."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    token_count: int = 0
    topic: Optional[str] = None
    user_intent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FallbackStrategy(ABC):
    """Base class for fallback strategies."""
    
    @abstractmethod
    def execute(self, context: ConversationContext) -> FallbackResponse:
        """
        Execute the fallback strategy.
        
        Args:
            context: Current conversation context
            
        Returns:
            FallbackResponse with action and message
        """
        pass


class ClarificationFallback(FallbackStrategy):
    """
    Request clarification instead of guessing.
    
    Best for: Confidence degradation, ambiguous requests.
    """
    
    DEFAULT_MESSAGE = "I need more specific information to help accurately."
    
    DEFAULT_SUGGESTIONS = [
        "Can you provide more details about the specific issue?",
        "What exactly are you trying to accomplish?",
        "Could you share any error messages or specific symptoms?",
    ]
    
    def __init__(
        self,
        message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        max_suggestions: int = 3,
    ):
        """
        Initialize the fallback.
        
        Args:
            message: Custom clarification message
            suggestions: Custom clarifying questions
            max_suggestions: Maximum number of suggestions to include
        """
        self.message = message or self.DEFAULT_MESSAGE
        self.suggestions = suggestions or self.DEFAULT_SUGGESTIONS
        self.max_suggestions = max_suggestions
    
    def execute(self, context: ConversationContext) -> FallbackResponse:
        """Request clarification from the user."""
        # Generate context-aware suggestions if possible
        suggestions = self._generate_suggestions(context)
        
        return FallbackResponse(
            action="request_clarification",
            message=self.message,
            suggestions=suggestions[:self.max_suggestions],
            metadata={"trigger": "confidence_degradation"},
        )
    
    def _generate_suggestions(self, context: ConversationContext) -> List[str]:
        """Generate context-aware clarifying questions."""
        # In production, this could use the conversation history
        # to generate more relevant questions
        
        suggestions = list(self.suggestions)
        
        if context.topic:
            suggestions.insert(0, f"What specific aspect of {context.topic} needs help?")
        
        return suggestions


class ContextResetFallback(FallbackStrategy):
    """
    Summarize and reset context when tokens run low.
    
    Best for: Context/token exhaustion.
    """
    
    DEFAULT_MESSAGE = "Let me summarize what we've covered so far..."
    
    def __init__(
        self,
        message: Optional[str] = None,
        summarizer: Optional[callable] = None,
    ):
        """
        Initialize the fallback.
        
        Args:
            message: Custom summary introduction
            summarizer: Optional custom summarization function
        """
        self.message = message or self.DEFAULT_MESSAGE
        self.summarizer = summarizer
    
    def execute(self, context: ConversationContext) -> FallbackResponse:
        """Summarize context and prepare for reset."""
        summary = self._generate_summary(context)
        
        return FallbackResponse(
            action="context_reset",
            message=self.message,
            suggestions=[],
            metadata={
                "trigger": "context_exhaustion",
                "summary": summary,
                "original_token_count": context.token_count,
            },
        )
    
    def _generate_summary(self, context: ConversationContext) -> str:
        """Generate a summary of the conversation."""
        if self.summarizer:
            return self.summarizer(context)
        
        # Simple extractive summary
        if not context.messages:
            return "No previous context to summarize."
        
        # Get key points (simplified)
        key_points = []
        for msg in context.messages[-5:]:  # Last 5 messages
            if msg.get("role") == "user":
                content = msg.get("content", "")[:100]
                key_points.append(f"- User asked about: {content}...")
        
        return "\n".join(key_points) if key_points else "Previous discussion context."


class EscalationFallback(FallbackStrategy):
    """
    Escalate to human agent or alternative system.
    
    Best for: Repeated failures, high-stakes decisions, explicit user request.
    """
    
    DEFAULT_MESSAGE = "I'm having difficulty with this request. Let me connect you with additional help."
    
    def __init__(
        self,
        message: Optional[str] = None,
        escalation_handler: Optional[callable] = None,
        include_context: bool = True,
    ):
        """
        Initialize the fallback.
        
        Args:
            message: Custom escalation message
            escalation_handler: Optional handler to process escalation
            include_context: Whether to include conversation context
        """
        self.message = message or self.DEFAULT_MESSAGE
        self.escalation_handler = escalation_handler
        self.include_context = include_context
    
    def execute(self, context: ConversationContext) -> FallbackResponse:
        """Escalate to human or alternative system."""
        # Call escalation handler if provided
        if self.escalation_handler:
            try:
                self.escalation_handler(context)
            except Exception as e:
                logger.error(f"Escalation handler failed: {e}")
        
        metadata = {"trigger": "escalation_requested"}
        
        if self.include_context:
            metadata["context_summary"] = self._summarize_for_handoff(context)
        
        return FallbackResponse(
            action="escalate",
            message=self.message,
            suggestions=["Would you like to speak with a human agent?"],
            metadata=metadata,
        )
    
    def _summarize_for_handoff(self, context: ConversationContext) -> Dict[str, Any]:
        """Create a summary for human handoff."""
        return {
            "topic": context.topic,
            "intent": context.user_intent,
            "message_count": len(context.messages),
            "last_messages": context.messages[-3:] if context.messages else [],
        }


class LimitedReasoningFallback(FallbackStrategy):
    """
    Continue with limited scope during HALF_OPEN recovery.
    
    Best for: Testing recovery after circuit trip.
    """
    
    DEFAULT_MESSAGE = "I'll try to help with a focused approach..."
    
    def __init__(
        self,
        message: Optional[str] = None,
        scope_limit: str = "single_step",
        token_budget: int = 500,
    ):
        """
        Initialize the fallback.
        
        Args:
            message: Custom message
            scope_limit: Scope restriction ("single_step", "direct_answer", etc.)
            token_budget: Maximum tokens for limited response
        """
        self.message = message or self.DEFAULT_MESSAGE
        self.scope_limit = scope_limit
        self.token_budget = token_budget
    
    def execute(self, context: ConversationContext) -> FallbackResponse:
        """Continue with limited scope."""
        return FallbackResponse(
            action="limited_reasoning",
            message=self.message,
            suggestions=[],
            metadata={
                "trigger": "half_open_recovery",
                "scope": self.scope_limit,
                "token_budget": self.token_budget,
            },
        )


# Convenience function
def get_fallback(
    circuit_state: str,
    **kwargs
) -> FallbackStrategy:
    """
    Get appropriate fallback for a circuit state.
    
    Args:
        circuit_state: Current circuit state
        **kwargs: Arguments passed to fallback constructor
        
    Returns:
        Appropriate FallbackStrategy instance
    """
    fallbacks = {
        "reasoning_open": ClarificationFallback,
        "context_open": ContextResetFallback,
        "half_open": LimitedReasoningFallback,
    }
    
    fallback_class = fallbacks.get(circuit_state, ClarificationFallback)
    return fallback_class(**kwargs)
