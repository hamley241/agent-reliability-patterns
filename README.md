# Agent Reliability Patterns

**Production-ready reliability patterns for AI agents, inspired by distributed systems.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

Your AI agent just burned through $47 worth of tokens chasing its own tail. Traditional circuit breakers catch network failures—but they can't catch **reasoning failures**.

This library applies proven distributed systems patterns to AI agent reliability:

- **Reasoning Circuit Breakers** — Detect confidence degradation before token waste
- **Context-Aware Load Shedding** — Handle concurrent users without context overflow
- **Confidence-Weighted Consensus** — Multi-agent decision validation
- **Credit-Based Backpressure** — Rate limiting based on reasoning quality

## Installation

```bash
pip install agent-reliability-patterns
```

Or from source:

```bash
git clone https://github.com/hamley241/agent-reliability-patterns.git
cd agent-reliability-patterns
pip install -e .
```

## Quick Start

```python
from agent_reliability import AIAgentCircuitBreaker, AgentResponse

# Initialize the circuit breaker
breaker = AIAgentCircuitBreaker(
    confidence_threshold=0.5,
    token_limit_percent=0.8
)

# In your agent loop
response = AgentResponse(
    text="I can help with that database issue...",
    confidence=0.85,
    token_count=150
)

if breaker.should_trip(response):
    # Pivot to clarification instead of continuing to guess
    return request_clarification(context)
else:
    # Continue normal reasoning
    continue_reasoning()
```

## Patterns Included

### 1. Reasoning Circuit Breaker

Detects reasoning failures that HTTP status codes miss:

| State | Trigger | Protection |
|-------|---------|------------|
| `REASONING_CLOSED` | Normal confidence | Default operation |
| `REASONING_OPEN` | Confidence drops | Stops token waste |
| `CONTEXT_OPEN` | Token limit hit | Prevents overflow |
| `HALF_OPEN` | Recovery test | Gradual re-engagement |

```python
from agent_reliability import AIAgentCircuitBreaker

breaker = AIAgentCircuitBreaker(
    confidence_threshold=0.5,      # Trip below 50% avg confidence
    token_limit_percent=0.8,       # Trip at 80% context usage
    recovery_timeout=30            # Seconds before HALF_OPEN
)
```

### 2. Confidence Measurement

Three approaches for measuring LLM confidence:

```python
from agent_reliability.confidence import (
    SecondaryModelEvaluator,
    SelfConsistencyChecker,
    LogitBasedHeuristics
)

# Recommended: Secondary model evaluation
evaluator = SecondaryModelEvaluator(model="gpt-3.5-turbo")
confidence = evaluator.evaluate(response_text)
```

### 3. Fallback Strategies

```python
from agent_reliability.fallbacks import (
    ClarificationFallback,
    ContextResetFallback,
    EscalationFallback
)

fallback = ClarificationFallback()
result = fallback.execute(context)
# Returns: "I need more specific information to help accurately."
```

## Benchmarks

Run the benchmark suite:

```bash
python3 -m benchmarks.customer_service
```

Early results from synthetic customer service scenarios:

| Metric | Without Breaker | With Breaker |
|--------|-----------------|--------------|
| Avg tokens/session | ~1,800 | ~1,200 |
| Context overflows | Common | Eliminated |
| Resolution quality | Variable | Improved |

See `/benchmarks` for methodology and reproducible scenarios.

## Configuration

```python
from agent_reliability import Config

config = Config(
    # Circuit breaker settings
    confidence_threshold=0.5,
    confidence_window_size=5,
    token_limit_percent=0.8,
    recovery_timeout_seconds=30,
    
    # Confidence measurement
    evaluator_model="gpt-3.5-turbo",
    consistency_samples=3,
    
    # Monitoring
    enable_prometheus=True,
    metrics_port=9090
)
```

## Monitoring

Prometheus metrics included:

```python
from agent_reliability.metrics import setup_metrics

setup_metrics(port=9090)

# Exposes:
# - agent_circuit_state{state="reasoning_closed|open|half_open"}
# - agent_confidence_score (histogram)
# - agent_tokens_used (counter)
# - agent_breaker_trips_total (counter)
```

## Background

This library applies patterns from:

- **Netflix Hystrix** — Circuit breaker state machines
- **AWS Lambda Health Checks** — Sliding window analysis
- **Kubernetes Probes** — Liveness and readiness separation

Adapted for AI-specific failure modes: confidence degradation, token exhaustion, hallucination spirals, and reasoning loops.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Goutham Patley** — 10 years building distributed systems at AWS (Lambda, BigMac). Now applying SRE patterns to AI agent reliability.

- [LinkedIn](https://linkedin.com/in/goutham-patley-b1391b41)
- [GitHub](https://github.com/hamley241)

---

*Part of the Agent Reliability Series. Next: Context-Aware Load Shedding.*
