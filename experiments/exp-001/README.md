# exp-001 Circuit Breaker Reliability Study

This directory contains the implementation and tools for exp-001, a comprehensive study of circuit breaker patterns for multi-agent AI systems reliability.

## Quick Start

### Run Local Simulator

The simulator allows you to iterate on circuit breaker logic and recovery policies locally before expensive Modal runs:

```bash
# Basic control workload test
python simulator.py --workload control --runs 10 --condition SIMPLE_CB --seed 42

# Verbose output to see per-turn events
python simulator.py --workload control --runs 5 --condition AI_CB --verbose

# Save results to file
python simulator.py --workload stress --runs 20 --condition ADAPTIVE_CB --output results.json
```

### Run Real Modal Experiment

```bash
# Pilot run (10 runs/condition)
python modal_app.py --runs 10

# Full experiment (55 runs/condition) 
python modal_app.py --full

# Real API mode (uses OpenAI/Anthropic APIs)
python modal_app.py --real --runs 5
```

## Simulator Features

The `simulator.py` provides a local test harness that mirrors the Modal experiment structure but runs with stubbed LLM responses for fast iteration:

### Key Features

- **Deterministic Testing**: Seeded random responses for reproducible results
- **Circuit Breaker Validation**: Test all protection modes (NO_PROTECTION → ADAPTIVE_CB)
- **Recovery Policy Testing**: Validate retry, skip, alternate-model, and safe-mode policies  
- **Workload Support**: Both control (green-path) and stress workloads
- **Detailed Logging**: Per-turn timeline events and recovery attempt tracking
- **Compatible Output**: JSON format matches Modal experiment results

### Protection Modes

| Condition      | Protection Mechanism           | Recovery Policy                              |
|----------------|--------------------------------|---------------------------------------------|
| NO_PROTECTION  | None                           | None                                        |
| TIMEOUT_ONLY   | Hard timeout on agent calls    | Retry once, then fail subtask              |
| SIMPLE_CB      | Fixed thresholds               | Skip failing agent, continue with plan     |
| AI_CB          | Reasoning-aware breaker        | Alternate prompt/model (Claude ↔ GPT-4o)   |
| ADAPTIVE_CB    | Predictive thresholds          | Safe-mode workflow with degraded objectives |

### Workloads

#### Control Workload (`workloads/control/`)
- **Purpose**: Validate infrastructure and recovery logic
- **Task**: Deterministic factual retrieval (capital of France)
- **Expected Success**: >95% unless protection interferes
- **Use Case**: Green-path testing, circuit breaker validation

#### Stress Workload (`workloads/stress/`)
- **Purpose**: Test protection under realistic failure conditions
- **Task**: Complex multi-agent routing scenario (simplified placeholder)
- **Expected Success**: ~30% without protection, higher with protection
- **Use Case**: Recovery policy validation, cascade prevention testing

## Configuration Options

### Simulator CLI Arguments

```bash
python simulator.py [OPTIONS]

Options:
  --workload {control,stress}    Workload type (default: control)
  --runs INT                     Number of runs (default: 10)  
  --condition {NO_PROTECTION,    Protection condition (default: SIMPLE_CB)
              TIMEOUT_ONLY,
              SIMPLE_CB,AI_CB,
              ADAPTIVE_CB}
  --seed INT                     Random seed for reproducibility (default: 42)
  --turn-budget INT              Turn budget per run (default: 25)
  --verbose, -v                  Verbose per-turn logging
  --output, -o FILE              Save results to JSON file
```

### Workload Configuration

Workloads are defined in `workloads/{name}/config.json`:

```json
{
  "failure_rates": {
    "api_timeout": 0.05,
    "confidence_decay": 0.03,
    "context_overflow": 0.02,
    "cascading_hallucination": 0.01
  },
  "agent_prompts": {
    "agent_a": "Prompt for agent A...",
    "agent_b": "Prompt for agent B..."
  },
  "expected_outputs": {
    "agent_a": "Expected response from agent A...",
    "agent_b": "Expected response from agent B..."  
  }
}
```

## Analysis and Results

### Output Format

Both simulator and Modal runs produce compatible JSON results:

```json
{
  "runs": 10,
  "avg_cfr": 0.1234,
  "completion_rate": 0.8500,
  "total_token_usage": 45000,
  "protection_activation_rate": 0.2000,
  "runs": [
    {
      "run_id": "SIMPLE_CB_control_001",
      "condition": "SIMPLE_CB", 
      "workload": "control",
      "task_completed": true,
      "metrics": { ... },
      "cfr": { ... },
      "timeline": [ ... ]
    }
  ]
}
```

### Key Metrics

- **CFR (Cascading Failure Rate)**: Percentage of initial failures that cascade
- **Completion Rate**: Percentage of tasks completed successfully
- **Protection Activation Rate**: How often circuit breakers trip
- **Recovery Success Rate**: How often recovery policies succeed
- **Token Efficiency**: Average tokens used per completed task

## Development Workflow

### Iteration Cycle

1. **Develop locally** with simulator for fast feedback:
   ```bash
   python simulator.py --workload control --runs 5 --condition AI_CB --verbose
   ```

2. **Validate with small Modal pilot**:
   ```bash
   python modal_app.py --runs 5
   ```

3. **Full Modal run** when confident:
   ```bash  
   python modal_app.py --full
   ```

### Debugging

Use verbose mode to see detailed per-turn events:
```bash
python simulator.py --workload control --runs 2 --condition SIMPLE_CB --verbose
```

Example output:
```
Turn 1: agent_call (agent_a) - {'turn': 1}
Turn 1: agent_success (agent_a) - {'confidence': 0.85, 'tokens': 1234}
Turn 2: agent_call (agent_b) - {'turn': 2}  
Turn 2: circuit_trip (agent_b) - {'error': 'Circuit breaker is OPEN'}
Turn 2: recovery_attempt (agent_b) - {'policy': 'skip_continue', 'error': '...'}
Turn 2: recovery_success (agent_b) - {'policy': 'SIMPLE_CB'}
```

## Files

- `simulator.py` - Local test harness for circuit breaker experimentation
- `experiment_runner.py` - Core experiment logic (used by Modal)
- `modal_app.py` - Modal deployment and CLI interface
- `circuit_breaker.py` - Circuit breaker implementations  
- `api_clients.py` - Real API client wrappers
- `workloads/control/` - Green-path control workload
- `workloads/stress/` - Stress testing workload
- `exp-001b-plan.md` - Detailed experiment plan and objectives

## Next Steps

1. **Validate simulator** with control workload across all conditions
2. **Implement full stress workload** with ≥25 turns and recovery hooks
3. **Run diagnostics** (2 runs per condition) to validate failure modes
4. **Execute full Modal sweep** (10 runs per condition)
5. **Analyze results** and prepare publication materials

For detailed objectives and analysis charter, see `exp-001b-plan.md`.