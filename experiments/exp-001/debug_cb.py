#!/usr/bin/env python3
"""Debug circuit breaker integration"""

from experiment_runner import SimulatedMultiAgentSystem

def debug_single_run():
    """Run a single task and inspect circuit breaker metrics."""
    system = SimulatedMultiAgentSystem("AI_CB", "debug_001")
    result = system.run_task()
    
    print("=== SINGLE RUN DEBUG ===")
    print(f"Condition: {result['condition']}")
    print(f"Task completed: {result['task_completed']}")
    print(f"CFR: {result['cfr']}")
    print(f"Metrics: {result['metrics']}")
    print(f"CB A: {result.get('circuit_breaker_a', 'MISSING')}")
    print(f"CB B: {result.get('circuit_breaker_b', 'MISSING')}")
    
    # Check if CBs have actual state
    if 'circuit_breaker_a' in result:
        cb_a = result['circuit_breaker_a']
        print(f"CB A State: {cb_a.get('state', 'UNKNOWN')}")
        print(f"CB A Trip Count: {cb_a.get('trip_count', 0)}")
        print(f"CB A Failures: {cb_a.get('consecutive_failures', 0)}")
    
    return result

if __name__ == "__main__":
    debug_single_run()