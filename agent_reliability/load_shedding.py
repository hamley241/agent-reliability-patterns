"""
Context-Aware Load Shedding for AI Agents
=========================================

Proactive load shedding with pre-flight estimation, priority awareness,
and graduated degradation. Works with circuit breakers for defense in depth.
"""

import time
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any
from collections import deque


class Priority(Enum):
    """Task priorities for load shedding"""
    CRITICAL = auto()
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()
    BACKGROUND = auto()


class DegradationLevel(Enum):
    """Graduated degradation steps"""
    FULL = "full"
    SIMPLIFIED = "simplified"
    CACHED = "cached"
    MINIMAL = "minimal"
    REJECT = "rejected"


@dataclass
class Task:
    """Task to be processed"""
    id: str
    prompt: str
    priority: Priority

    def get_complexity(self) -> float:
        """Estimate reasoning complexity (0-1)"""
        complexity = 0.5
        if len(self.prompt) > 500:
            complexity += 0.2
        if 'analyze' in self.prompt.lower() or 'step by step' in self.prompt.lower():
            complexity += 0.2
        return min(complexity, 1.0)

    def estimate_tokens(self) -> Dict[str, Any]:
        """Pre-flight token estimation"""
        prompt_tokens = len(self.prompt) // 4
        complexity = self.get_complexity()
        output_tokens = int(500 * (1 + complexity * 3))
        total = prompt_tokens + output_tokens
        return {
            'total': total,
            'complexity': complexity,
            'min': int(total * 0.7),
            'max': int(total * 1.3)
        }


@dataclass
class TaskResult:
    """Result of task processing"""
    task_id: str
    success: bool
    tokens_used: int
    degradation: DegradationLevel
    latency_ms: float


class LoadShedder:
    """Proactive load shedder for AI systems"""

    def __init__(self, budget: int = 10000):
        self.budget = budget
        self.tokens_used = 0
        self.shed_count = 0
        self.degraded_count = 0

    def get_load(self) -> float:
        """Current load level (0-1)"""
        return min(self.tokens_used / (self.budget * 0.9), 1.0)

    def get_remaining(self) -> int:
        """Remaining budget"""
        return max(0, int(self.budget * 0.9) - self.tokens_used)

    def submit(self, task: Task) -> TaskResult:
        """Submit task with load shedding"""
        estimate = task.estimate_tokens()
        load = self.get_load()
        remaining = self.get_remaining()

        print(f"\n📥 {task.id} [{task.priority.name}]: {task.prompt[:40]}...")
        print(f"   💰 Estimated: {estimate['total']} tokens (complexity: {estimate['complexity']:.2f})")
        print(f"   📊 Load: {load*100:.0f}%, Remaining: {remaining}")

        # Priority-based shedding
        if task.priority == Priority.BACKGROUND and load > 0.3:
            self.shed_count += 1
            print(f"   🚫 REJECTED: Background task at {load*100:.0f}% load")
            return TaskResult(task.id, False, 0, DegradationLevel.REJECT, 0)

        if task.priority == Priority.LOW and load > 0.5:
            self.shed_count += 1
            print(f"   🚫 REJECTED: Low priority at {load*100:.0f}% load")
            return TaskResult(task.id, False, 0, DegradationLevel.REJECT, 0)

        # Budget-based degradation
        if estimate['total'] > remaining * 0.9:
            return self._process(task, DegradationLevel.MINIMAL, estimate)
        elif estimate['total'] > remaining * 0.7:
            return self._process(task, DegradationLevel.CACHED, estimate)
        elif estimate['total'] > remaining * 0.5:
            return self._process(task, DegradationLevel.SIMPLIFIED, estimate)
        else:
            return self._process(task, DegradationLevel.FULL, estimate)

    def _process(self, task: Task, level: DegradationLevel, estimate: Dict) -> TaskResult:
        """Process task with given degradation level"""
        if level == DegradationLevel.FULL:
            tokens = estimate['total']
            print(f"   ✅ FULL: {tokens} tokens")
        elif level == DegradationLevel.SIMPLIFIED:
            tokens = int(estimate['total'] * 0.7)
            self.degraded_count += 1
            print(f"   🟡 SIMPLIFIED: {tokens} tokens (30% reduction)")
        elif level == DegradationLevel.CACHED:
            tokens = 100
            self.degraded_count += 1
            print(f"   📦 CACHED: {tokens} tokens (cached response)")
        elif level == DegradationLevel.MINIMAL:
            tokens = 50
            self.degraded_count += 1
            print(f"   🔻 MINIMAL: {tokens} tokens (minimal response)")

        self.tokens_used += tokens
        return TaskResult(task.id, True, tokens, level, random.randint(50, 500))

    def get_stats(self) -> Dict[str, Any]:
        return {
            'load': f"{self.get_load()*100:.1f}%",
            'remaining': self.get_remaining(),
            'shed': self.shed_count,
            'degraded': self.degraded_count,
            'total_used': self.tokens_used
        }


# Demo
if __name__ == "__main__":
    print("🔬 Context-Aware Load Shedding Demo")
    print("=" * 50)

    shedder = LoadShedder(budget=5000)

    # Simulate incoming tasks
    tasks = [
        Task("t1", "Summarize this document briefly", Priority.NORMAL),
        Task("t2", "Analyze step by step the market trends for Q4", Priority.HIGH),
        Task("t3", "Background cleanup task", Priority.BACKGROUND),
        Task("t4", "Explain quantum computing principles in detail", Priority.CRITICAL),
        Task("t5", "Generate daily report summary", Priority.LOW),
        Task("t6", "Deep dive analysis with step by step reasoning on AI ethics", Priority.HIGH),
        Task("t7", "User query: what is 2+2?", Priority.CRITICAL),
        Task("t8", "Background indexing task", Priority.BACKGROUND),
        Task("t9", "Compare and evaluate multiple strategies for optimization", Priority.NORMAL),
        Task("t10", "System health check", Priority.BACKGROUND),
    ]

    print(f"\nToken budget: {shedder.budget}")
    print("Processing tasks...")

    results = []
    for task in tasks:
        result = shedder.submit(task)
        results.append(result)
        time.sleep(0.1)

    print(f"\n" + "=" * 50)
    print("📊 STATS:")
    stats = shedder.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print(f"\n✅ Completed: {len([r for r in results if r.success])}/{len(results)}")
    print(f"🚫 Shed: {len([r for r in results if r.degradation == DegradationLevel.REJECT])}")
    print(f"🟡 Degraded: {len([r for r in results if r.success and r.degradation != DegradationLevel.FULL])}")