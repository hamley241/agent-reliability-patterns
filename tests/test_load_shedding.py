"""Tests for the Load Shedding pattern."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'patterns', '02-load-shedding'))

class LoadShedder:
    """Simplified load shedder for testing."""
    def __init__(self, max_tokens=10000, priority_levels=None):
        self.max_tokens = max_tokens
        self.priority_levels = priority_levels or {
            "CRITICAL": 1.0,
            "HIGH": 0.8,
            "NORMAL": 0.6,
            "LOW": 0.4
        }
        self.current_load = 0
        self.shed_count = 0
        self.accepted_count = 0
    
    def estimate_tokens(self, text):
        return len(text) // 4  # Simple estimation
    
    def can_accept(self, task_tokens, priority="NORMAL"):
        utilization = self.current_load / self.max_tokens
        threshold = self.priority_levels.get(priority, 0.6)
        return utilization <= threshold
    
    def submit(self, task_text, priority="NORMAL"):
        tokens = self.estimate_tokens(task_text)
        if not self.can_accept(tokens, priority):
            self.shed_count += 1
            raise Exception(f"Load shed: priority={priority}")
        self.current_load += tokens
        self.accepted_count += 1
        return True
    
    def complete(self, task_text):
        tokens = self.estimate_tokens(task_text)
        self.current_load -= tokens

class TestLoadShedder:
    """Test cases for load shedding functionality."""
    
    def test_accepts_normal_priority_under_capacity(self):
        shedder = LoadShedder(max_tokens=1000)
        assert shedder.submit("simple task", priority="NORMAL") is True
        assert shedder.current_load == 3  # "simple task" = 10 chars / 4 = 3 tokens
    
    def test_sheds_low_priority_when_utilized(self):
        shedder = LoadShedder(max_tokens=100)
        # Fill to 60% of capacity
        shedder.current_load = 50  # 50% utilized
        # Low priority should shed at 40%
        with pytest.raises(Exception) as exc_info:
            shedder.submit("low priority task", priority="LOW")
        assert "Load shed" in str(exc_info.value)
        assert shedder.shed_count == 1
    
    def test_critical_always_accepted(self):
        shedder = LoadShedder(max_tokens=100)
        shedder.current_load = 95  # 95% utilized (critical can go to 100%)
        # Critical priority should still be accepted
        assert shedder.submit("critical task", priority="CRITICAL") is True
        assert shedder.accepted_count == 1
    
    def test_complexity_based_shedding(self):
        shedder = LoadShedder(max_tokens=100)
        # Simple task should be accepted
        assert shedder.submit("short", priority="NORMAL") is True
        # Complex task should also be accepted if under threshold
        complex_task = "analyze this" * 10  # 100 chars = 25 tokens
        assert shedder.submit(complex_task, priority="NORMAL") is True
    
    def test_shedding_protects_system(self):
        shedder = LoadShedder(max_tokens=100)
        shedder.current_load = 70  # 70% utilized (above normal threshold)
        # Normal priority should be shed
        with pytest.raises(Exception):
            shedder.submit("normal task", priority="NORMAL")
        # But high priority should be accepted
        assert shedder.submit("high task", priority="HIGH") is True
    
    def test_load_released_on_completion(self):
        shedder = LoadShedder(max_tokens=1000)
        task = "test task"
        tokens = shedder.estimate_tokens(task)
        
        shedder.submit(task, priority="NORMAL")
        assert shedder.current_load == tokens
        
        shedder.complete(task)
        assert shedder.current_load == 0

class TestPriorityHierarchy:
    """Test priority-based shedding hierarchy."""
    
    def test_priority_order(self):
        shedder = LoadShedder(max_tokens=100)
        shedder.current_load = 65  # 65% utilized
        
        # Normal should shed at 60%
        with pytest.raises(Exception):
            shedder.submit("normal", "NORMAL")
        
        # High should accept at 80%
        assert shedder.submit("high", "HIGH") is True
    
    def test_critical_always_works(self):
        shedder = LoadShedder(max_tokens=100)
        # Even at 99% capacity
        shedder.current_load = 99
        # Critical still works
        assert shedder.submit("critical", "CRITICAL") is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])