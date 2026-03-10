"""
Basic Usage Example - AI Agent Circuit Breaker

This example shows how to integrate the circuit breaker into
an AI agent's reasoning loop.
"""

from agent_reliability.circuit_breaker import AIAgentCircuitBreaker, AgentResponse, CircuitState
from agent_reliability.confidence import SecondaryModelEvaluator
from agent_reliability.fallbacks import (
    ClarificationFallback,
    ContextResetFallback,
    get_fallback,
)


def simulate_agent_response(query: str, turn: int) -> tuple[str, float, int]:
    """
    Simulate an AI agent generating a response.
    
    In production, this would be your actual LLM call.
    Returns: (response_text, confidence, token_count)
    """
    # Simulate degrading confidence on complex queries
    if "complex" in query.lower():
        confidence = max(0.2, 0.9 - (turn * 0.15))
    else:
        confidence = 0.85
    
    responses = [
        "Let me analyze that for you...",
        "Based on my analysis...",
        "Looking at this more closely...",
        "Actually, let me reconsider...",  # This will trigger circular detection
        "I think the answer might be...",
    ]
    
    response = responses[min(turn, len(responses) - 1)]
    tokens = len(response.split()) * 4  # Rough estimate
    
    return response, confidence, tokens


def main():
    """Demonstrate circuit breaker in action."""
    
    # Initialize circuit breaker
    breaker = AIAgentCircuitBreaker(
        confidence_threshold=0.5,
        token_limit_percent=0.8,
        recovery_timeout=30,
        max_context_tokens=4096,
    )
    
    # Set up callbacks for monitoring
    def on_state_change(old_state, new_state):
        print(f"  [Circuit] State changed: {old_state.value} -> {new_state.value}")
    
    def on_trip(state, reason):
        print(f"  [Circuit] TRIPPED! State: {state.value}, Reason: {reason}")
    
    breaker.on_state_change(on_state_change)
    breaker.on_trip(on_trip)
    
    # Initialize fallbacks
    clarification_fallback = ClarificationFallback()
    context_reset_fallback = ContextResetFallback()
    
    # Simulate a conversation
    queries = [
        "What is the capital of France?",  # Simple - should pass
        "Explain complex database optimization",  # Complex - will degrade
        "More details on complex query optimization",  # Continued complexity
        "Even more complex distributed systems",  # Will likely trip
    ]
    
    print("=" * 60)
    print("AI Agent Circuit Breaker Demo")
    print("=" * 60)
    
    for i, query in enumerate(queries):
        print(f"\n[Turn {i + 1}] User: {query}")
        
        # Check if circuit allows requests
        if breaker.is_open:
            print("  [Circuit] Circuit is OPEN - checking for recovery...")
            if breaker.attempt_recovery():
                print("  [Circuit] Attempting recovery in HALF_OPEN state")
            else:
                # Execute fallback
                fallback = get_fallback(breaker.state.value)
                result = fallback.execute(None)
                print(f"  [Fallback] {result.message}")
                continue
        
        # Simulate agent response
        response_text, confidence, tokens = simulate_agent_response(query, i)
        
        # Create response object
        response = AgentResponse(
            text=response_text,
            confidence=confidence,
            token_count=tokens,
        )
        
        print(f"  [Agent] Response: {response_text}")
        print(f"  [Metrics] Confidence: {confidence:.2f}, Tokens: {tokens}")
        
        # Check if circuit should trip
        if breaker.should_trip(response):
            # Get appropriate fallback
            fallback = get_fallback(breaker.state.value)
            result = fallback.execute(None)
            print(f"  [Fallback] {result.message}")
            if result.suggestions:
                print(f"  [Suggestions] {result.suggestions[0]}")
        else:
            print("  [Circuit] Request successful, circuit closed")
    
    # Print final stats
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    stats = breaker.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
