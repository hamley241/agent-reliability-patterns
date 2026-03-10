"""
Confidence Measurement for AI Agents

Three approaches for measuring LLM response confidence:
1. Secondary Model Evaluation - Use a lightweight model to evaluate
2. Self-Consistency Checks - Sample multiple responses, measure agreement
3. Logit-Based Heuristics - Use token probability distributions
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceEvaluator(ABC):
    """Base class for confidence evaluation strategies."""
    
    @abstractmethod
    def evaluate(self, response_text: str, **kwargs) -> float:
        """
        Evaluate confidence of a response.
        
        Args:
            response_text: The text to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass


class SecondaryModelEvaluator(ConfidenceEvaluator):
    """
    Use a lightweight model to evaluate the primary model's output.
    
    This is the recommended approach - most reliable and doesn't
    require access to model internals.
    
    Latency consideration: Adds 100-300ms per request.
    """
    
    EVALUATION_PROMPT = """Rate the confidence level (0.0-1.0) of this response.
Consider:
- Specificity: Are claims specific or vague?
- Internal consistency: Does it contradict itself?
- Hedge words: "might", "possibly", "I think" reduce confidence
- Certainty markers: "definitely", "certainly" increase confidence

Response to evaluate:
"{response}"

Return ONLY a number between 0.0 and 1.0."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        client: Optional[object] = None,
        timeout: float = 10.0,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to use for evaluation (lightweight recommended)
            client: Optional pre-configured API client
            timeout: Request timeout in seconds
        """
        self.model = model
        self.client = client
        self.timeout = timeout
    
    def evaluate(self, response_text: str, **kwargs) -> float:
        """
        Evaluate confidence using secondary model.
        
        Args:
            response_text: The text to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if self.client is None:
            # Fallback to heuristic if no client configured
            logger.warning("No API client configured, using heuristic fallback")
            return self._heuristic_fallback(response_text)
        
        try:
            prompt = self.EVALUATION_PROMPT.format(response=response_text[:1000])
            
            # This is a placeholder - actual implementation depends on client
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[{"role": "user", "content": prompt}],
            #     max_tokens=10,
            #     temperature=0,
            # )
            # score = float(response.choices[0].message.content.strip())
            
            # For now, use heuristic
            score = self._heuristic_fallback(response_text)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Secondary model evaluation failed: {e}")
            return self._heuristic_fallback(response_text)
    
    def _heuristic_fallback(self, text: str) -> float:
        """Simple heuristic when API is unavailable."""
        text_lower = text.lower()
        
        # Negative signals (reduce confidence)
        hedge_words = ["might", "maybe", "possibly", "perhaps", "i think", "not sure"]
        hedge_count = sum(1 for word in hedge_words if word in text_lower)
        
        # Positive signals (increase confidence)
        certainty_words = ["definitely", "certainly", "clearly", "obviously", "always"]
        certainty_count = sum(1 for word in certainty_words if word in text_lower)
        
        # Base confidence
        confidence = 0.7
        confidence -= hedge_count * 0.1
        confidence += certainty_count * 0.05
        
        return max(0.1, min(0.95, confidence))


class SelfConsistencyChecker(ConfidenceEvaluator):
    """
    Sample multiple responses and measure agreement.
    
    Higher agreement = higher confidence.
    Useful for detecting when the model is uncertain.
    """
    
    def __init__(
        self,
        n_samples: int = 3,
        model: Optional[str] = None,
        client: Optional[object] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize the checker.
        
        Args:
            n_samples: Number of responses to sample
            model: Model to use for sampling
            client: Optional pre-configured API client
            temperature: Sampling temperature (higher = more variation)
        """
        self.n_samples = n_samples
        self.model = model
        self.client = client
        self.temperature = temperature
    
    def evaluate(self, response_text: str, prompt: Optional[str] = None, **kwargs) -> float:
        """
        Evaluate confidence via self-consistency.
        
        Args:
            response_text: The original response
            prompt: The original prompt (needed to generate alternatives)
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if prompt is None or self.client is None:
            logger.warning("Self-consistency requires prompt and client")
            return 0.5
        
        try:
            # Generate alternative responses
            responses = [response_text]
            
            # Placeholder for actual sampling
            # for _ in range(self.n_samples - 1):
            #     response = self.client.chat.completions.create(
            #         model=self.model,
            #         messages=[{"role": "user", "content": prompt}],
            #         temperature=self.temperature,
            #     )
            #     responses.append(response.choices[0].message.content)
            
            # Calculate semantic similarity
            similarity_scores = self._calculate_similarity(responses)
            
            return float(np.mean(similarity_scores))
            
        except Exception as e:
            logger.error(f"Self-consistency check failed: {e}")
            return 0.5
    
    def _calculate_similarity(self, responses: List[str]) -> List[float]:
        """Calculate pairwise semantic similarity."""
        # Placeholder - in production, use embeddings
        # For now, use simple word overlap as proxy
        
        if len(responses) < 2:
            return [1.0]
        
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self._word_overlap_similarity(responses[i], responses[j])
                similarities.append(sim)
        
        return similarities if similarities else [0.5]
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)


class LogitBasedHeuristics(ConfidenceEvaluator):
    """
    Use token probability distributions for confidence.
    
    Requires access to model logits (not available from all providers).
    High entropy = low confidence.
    """
    
    def __init__(self, max_entropy: float = 10.0):
        """
        Initialize the heuristics calculator.
        
        Args:
            max_entropy: Maximum expected entropy for normalization
        """
        self.max_entropy = max_entropy
    
    def evaluate(
        self,
        response_text: str,
        logits: Optional[List[float]] = None,
        **kwargs
    ) -> float:
        """
        Evaluate confidence from logits.
        
        Args:
            response_text: The response text (unused, for interface consistency)
            logits: Token logits from the model
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if logits is None:
            logger.warning("No logits provided, returning default confidence")
            return 0.5
        
        try:
            # Convert logits to probabilities
            logits_array = np.array(logits)
            probs = self._softmax(logits_array)
            
            # Calculate entropy
            entropy = self._calculate_entropy(probs)
            
            # Normalize to confidence score
            # High entropy = low confidence
            confidence = 1.0 - (entropy / self.max_entropy)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Logit-based confidence failed: {e}")
            return 0.5
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
    
    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Calculate Shannon entropy."""
        # Filter out zero probabilities to avoid log(0)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))


# Convenience function
def get_evaluator(method: str = "secondary", **kwargs) -> ConfidenceEvaluator:
    """
    Get a confidence evaluator by method name.
    
    Args:
        method: One of "secondary", "consistency", "logits"
        **kwargs: Arguments passed to evaluator constructor
        
    Returns:
        Configured ConfidenceEvaluator instance
    """
    evaluators = {
        "secondary": SecondaryModelEvaluator,
        "consistency": SelfConsistencyChecker,
        "logits": LogitBasedHeuristics,
    }
    
    if method not in evaluators:
        raise ValueError(f"Unknown method: {method}. Choose from {list(evaluators.keys())}")
    
    return evaluators[method](**kwargs)
