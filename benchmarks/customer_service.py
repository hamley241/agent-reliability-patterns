"""
Customer Service Benchmark

Synthetic benchmark demonstrating circuit breaker effectiveness
on simulated customer service conversations.

Run: python -m benchmarks.customer_service

This generates synthetic scenarios to measure:
- Token usage with/without circuit breakers
- Context overflow prevention
- Resolution quality (simulated)
"""

import random
import sys
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuit_breaker import AIAgentCircuitBreaker, AgentResponse, CircuitState


@dataclass
class BenchmarkResult:
    """Results from a single scenario run."""
    scenario_id: int
    tokens_used: int
    turns: int
    resolved: bool
    context_overflow: bool
    final_confidence: float
    breaker_tripped: bool


@dataclass
class AggregateResults:
    """Aggregated benchmark results."""
    total_scenarios: int
    avg_tokens: float
    context_overflows: int
    resolution_rate: float
    avg_confidence: float
    breaker_trips: int


class CustomerServiceScenario:
    """
    Simulates a customer service conversation.
    
    Scenarios have varying complexity that affects:
    - How quickly confidence degrades
    - How many tokens are consumed
    - Whether the issue gets resolved
    """
    
    COMPLEXITY_PROFILES = {
        "simple": {"confidence_decay": 0.02, "base_tokens": 50, "resolve_threshold": 0.6},
        "moderate": {"confidence_decay": 0.08, "base_tokens": 100, "resolve_threshold": 0.5},
        "complex": {"confidence_decay": 0.15, "base_tokens": 150, "resolve_threshold": 0.4},
        "adversarial": {"confidence_decay": 0.25, "base_tokens": 200, "resolve_threshold": 0.3},
    }
    
    def __init__(self, scenario_id: int, complexity: str = None):
        self.scenario_id = scenario_id
        self.complexity = complexity or random.choice(list(self.COMPLEXITY_PROFILES.keys()))
        self.profile = self.COMPLEXITY_PROFILES[self.complexity]
        self.current_confidence = 0.9 + random.uniform(-0.05, 0.05)
        self.tokens_used = 0
        self.turns = 0
        self.resolved = False
        
    def generate_turn(self) -> Tuple[str, float, int]:
        """
        Generate a simulated agent turn.
        
        Returns: (response_text, confidence, tokens)
        """
        self.turns += 1
        
        # Confidence degrades with complexity
        decay = self.profile["confidence_decay"] * (1 + random.uniform(-0.3, 0.3))
        self.current_confidence = max(0.1, self.current_confidence - decay)
        
        # Token usage varies
        tokens = int(self.profile["base_tokens"] * (1 + random.uniform(-0.2, 0.4)))
        self.tokens_used += tokens
        
        # Simulate response text
        if self.current_confidence > 0.7:
            text = f"I can help with that. Let me analyze the issue..."
        elif self.current_confidence > 0.5:
            text = f"Looking into this further..."
        elif self.current_confidence > 0.3:
            text = f"This is more complex than expected. Let me reconsider..."
        else:
            text = f"Actually, I was wrong about that. Let me try again..."
        
        return text, self.current_confidence, tokens
    
    def check_resolution(self) -> bool:
        """Check if scenario resolved successfully."""
        # Resolution more likely with higher confidence
        if self.current_confidence >= self.profile["resolve_threshold"]:
            self.resolved = random.random() < (self.current_confidence * 0.8)
        return self.resolved


def run_scenario_without_breaker(scenario: CustomerServiceScenario, max_turns: int = 10, max_tokens: int = 2000) -> BenchmarkResult:
    """Run scenario without circuit breaker protection."""
    context_overflow = False
    
    for _ in range(max_turns):
        text, confidence, tokens = scenario.generate_turn()
        
        # Check for context overflow
        if scenario.tokens_used > max_tokens:
            context_overflow = True
            break
        
        # Check if resolved
        if scenario.check_resolution():
            break
    
    return BenchmarkResult(
        scenario_id=scenario.scenario_id,
        tokens_used=scenario.tokens_used,
        turns=scenario.turns,
        resolved=scenario.resolved,
        context_overflow=context_overflow,
        final_confidence=scenario.current_confidence,
        breaker_tripped=False,
    )


def run_scenario_with_breaker(scenario: CustomerServiceScenario, max_turns: int = 10, max_tokens: int = 2000) -> BenchmarkResult:
    """Run scenario with circuit breaker protection."""
    breaker = AIAgentCircuitBreaker(
        confidence_threshold=0.5,
        token_limit_percent=0.8,
        max_context_tokens=max_tokens,
    )
    
    breaker_tripped = False
    context_overflow = False
    
    for _ in range(max_turns):
        text, confidence, tokens = scenario.generate_turn()
        
        response = AgentResponse(
            text=text,
            confidence=confidence,
            token_count=tokens,
        )
        
        if breaker.should_trip(response):
            breaker_tripped = True
            # Simulate fallback: request clarification (saves tokens, may resolve)
            scenario.current_confidence = min(0.7, scenario.current_confidence + 0.2)
            scenario.tokens_used += 30  # Small clarification response
            if random.random() < 0.6:  # Clarification often helps
                scenario.resolved = True
            break
        
        if scenario.check_resolution():
            break
    
    # Context overflow only if we didn't trip the breaker
    if scenario.tokens_used > max_tokens and not breaker_tripped:
        context_overflow = True
    
    return BenchmarkResult(
        scenario_id=scenario.scenario_id,
        tokens_used=scenario.tokens_used,
        turns=scenario.turns,
        resolved=scenario.resolved,
        context_overflow=context_overflow,
        final_confidence=scenario.current_confidence,
        breaker_tripped=breaker_tripped,
    )


def aggregate_results(results: List[BenchmarkResult]) -> AggregateResults:
    """Aggregate results from multiple scenarios."""
    return AggregateResults(
        total_scenarios=len(results),
        avg_tokens=sum(r.tokens_used for r in results) / len(results),
        context_overflows=sum(1 for r in results if r.context_overflow),
        resolution_rate=sum(1 for r in results if r.resolved) / len(results),
        avg_confidence=sum(r.final_confidence for r in results) / len(results),
        breaker_trips=sum(1 for r in results if r.breaker_tripped),
    )


def run_benchmark(n_scenarios: int = 100, seed: int = 42) -> Tuple[AggregateResults, AggregateResults]:
    """
    Run full benchmark comparing with/without circuit breakers.
    
    Args:
        n_scenarios: Number of scenarios to run
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (results_without_breaker, results_with_breaker)
    """
    random.seed(seed)
    
    # Run without breaker
    results_without = []
    for i in range(n_scenarios):
        scenario = CustomerServiceScenario(scenario_id=i)
        result = run_scenario_without_breaker(scenario)
        results_without.append(result)
    
    # Reset seed for fair comparison
    random.seed(seed)
    
    # Run with breaker
    results_with = []
    for i in range(n_scenarios):
        scenario = CustomerServiceScenario(scenario_id=i)
        result = run_scenario_with_breaker(scenario)
        results_with.append(result)
    
    return aggregate_results(results_without), aggregate_results(results_with)


def main():
    """Run benchmark and print results."""
    print("=" * 60)
    print("Customer Service Benchmark")
    print("Circuit Breaker for AI Agents")
    print("=" * 60)
    print()
    
    n_scenarios = 100
    print(f"Running {n_scenarios} synthetic customer service scenarios...")
    print()
    
    without_breaker, with_breaker = run_benchmark(n_scenarios=n_scenarios)
    
    # Calculate improvements
    token_reduction = (without_breaker.avg_tokens - with_breaker.avg_tokens) / without_breaker.avg_tokens * 100
    resolution_improvement = (with_breaker.resolution_rate - without_breaker.resolution_rate) / without_breaker.resolution_rate * 100
    
    print("Results:")
    print("-" * 60)
    print(f"{'Metric':<30} {'Without Breaker':<15} {'With Breaker':<15}")
    print("-" * 60)
    print(f"{'Avg tokens/session':<30} {without_breaker.avg_tokens:<15.0f} {with_breaker.avg_tokens:<15.0f}")
    print(f"{'Context overflows':<30} {without_breaker.context_overflows:<15} {with_breaker.context_overflows:<15}")
    print(f"{'Resolution rate':<30} {without_breaker.resolution_rate*100:<14.1f}% {with_breaker.resolution_rate*100:<14.1f}%")
    print(f"{'Avg final confidence':<30} {without_breaker.avg_confidence:<15.2f} {with_breaker.avg_confidence:<15.2f}")
    print("-" * 60)
    print()
    print(f"Token reduction: {token_reduction:.1f}%")
    print(f"Resolution improvement: {resolution_improvement:.1f}%")
    print(f"Context overflows eliminated: {without_breaker.context_overflows} -> {with_breaker.context_overflows}")
    print(f"Breaker trips: {with_breaker.breaker_trips}/{n_scenarios} scenarios")
    print()
    print("Note: This is a synthetic benchmark with simulated scenarios.")
    print("Real-world results will vary based on model, prompts, and use case.")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
