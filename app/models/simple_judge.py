"""
Simple Working LLM Judge Implementation
Uses sentence transformers for semantic similarity and rule-based scoring.
"""

import torch
import time
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleJudge:
    """
    Simple but effective LLM Judge using sentence transformers for semantic evaluation.
    This approach is reliable, fast, and doesn't require complex model dependencies.
    """
    
    def __init__(self):
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"  # Small, fast, reliable sentence transformer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
        logger.info(f"SimpleJudge initialized - Device: {self.device}")
    
    def _load_model(self):
        """Load the sentence transformer model (lazy loading)"""
        if self.model_loaded:
            return
            
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            start_time = time.time()
            
            # Load sentence transformer - this is much more reliable
            self.model = SentenceTransformer(self.model_name)
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"Simple Judge model loaded successfully in {load_time:.2f} seconds")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _calculate_relevance(self, prompt: str, response: str) -> float:
        """Calculate relevance score using semantic similarity"""
        if not self.model_loaded:
            self._load_model()
        
        try:
            # Get embeddings for prompt and response
            embeddings = self.model.encode([prompt, response])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Convert to 1-10 scale (similarity is 0-1, we want 1-10)
            score = 1 + (similarity * 9)  # Maps 0->1, 1->10
            
            return max(1.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Relevance calculation failed: {e}")
            return 5.0  # Default middle score
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence based on text structure and flow"""
        try:
            # Simple coherence metrics
            sentences = response.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) == 0:
                return 1.0
            
            score = 5.0  # Base score
            
            # Bonus for reasonable length
            word_count = len(response.split())
            if 20 <= word_count <= 200:
                score += 1.5
            elif word_count > 10:
                score += 0.5
            
            # Bonus for proper sentence structure
            if len(sentences) >= 2:
                score += 1.0
            
            # Bonus for proper capitalization and punctuation
            if response[0].isupper() if response else False:
                score += 0.5
            
            # Check for logical connectors
            connectors = ['therefore', 'however', 'furthermore', 'additionally', 'moreover', 'consequently']
            if any(connector in response.lower() for connector in connectors):
                score += 1.0
            
            return max(1.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {e}")
            return 5.0
    
    def _calculate_accuracy(self, prompt: str, response: str) -> float:
        """Calculate accuracy based on response completeness and factual indicators"""
        try:
            score = 5.0  # Base score
            
            # Check for specific, detailed information
            if len(response.split()) > 30:
                score += 1.5  # Detailed responses likely more accurate
            
            # Check for numbers, statistics, or specific facts
            if re.search(r'\d+', response):
                score += 1.0
            
            # Check for examples or specific instances
            example_words = ['example', 'instance', 'such as', 'including', 'like']
            if any(word in response.lower() for word in example_words):
                score += 1.0
            
            # Check for hedging (might indicate uncertainty but also carefulness)
            hedge_words = ['might', 'could', 'may', 'possibly', 'likely', 'typically']
            if any(word in response.lower() for word in hedge_words):
                score += 0.5
            
            # Penalty for very vague responses
            vague_words = ['thing', 'stuff', 'something', 'somehow']
            if any(word in response.lower() for word in vague_words):
                score -= 1.0
            
            return max(1.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return 5.0
    
    def _calculate_completeness(self, prompt: str, response: str) -> float:
        """Calculate completeness based on response thoroughness"""
        try:
            score = 5.0  # Base score
            
            # Length-based completeness
            word_count = len(response.split())
            if word_count > 100:
                score += 2.0
            elif word_count > 50:
                score += 1.5
            elif word_count > 20:
                score += 1.0
            elif word_count < 10:
                score -= 2.0
            
            # Check for structured responses (lists, sections)
            if re.search(r'\d+\.|\n[-*]|\n\d+\)', response):
                score += 1.5
            
            # Check for comprehensive coverage
            question_words = ['what', 'how', 'why', 'when', 'where', 'who']
            if any(word in prompt.lower() for word in question_words):
                # Check if response addresses multiple aspects
                aspects = response.count('.') + response.count('?') + response.count('\n')
                if aspects > 3:
                    score += 1.0
            
            return max(1.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Completeness calculation failed: {e}")
            return 5.0
    
    def evaluate(self, prompt: str, response: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation using simple but effective metrics
        """
        if weights is None:
            weights = {
                "relevance": 0.3,
                "coherence": 0.25,
                "accuracy": 0.25,
                "completeness": 0.2
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        start_time = time.time()
        
        try:
            # Calculate individual scores
            relevance_score = self._calculate_relevance(prompt, response)
            coherence_score = self._calculate_coherence(response)
            accuracy_score = self._calculate_accuracy(prompt, response)
            completeness_score = self._calculate_completeness(prompt, response)
            
            # Store detailed scores
            detailed_scores = {
                "relevance": round(relevance_score, 2),
                "coherence": round(coherence_score, 2), 
                "accuracy": round(accuracy_score, 2),
                "completeness": round(completeness_score, 2)
            }
            
            # Calculate weighted overall score
            overall_score = (
                relevance_score * weights["relevance"] +
                coherence_score * weights["coherence"] +
                accuracy_score * weights["accuracy"] +
                completeness_score * weights["completeness"]
            )
            
            # Generate detailed feedback
            detailed_feedback = {
                "relevance": f"Semantic similarity with prompt: {relevance_score:.1f}/10. Response {'closely matches' if relevance_score > 7 else 'somewhat relates to' if relevance_score > 4 else 'poorly matches'} the prompt.",
                "coherence": f"Text structure and flow: {coherence_score:.1f}/10. Response is {'well-structured' if coherence_score > 7 else 'adequately structured' if coherence_score > 4 else 'poorly structured'}.",
                "accuracy": f"Factual content assessment: {accuracy_score:.1f}/10. Response shows {'high' if accuracy_score > 7 else 'moderate' if accuracy_score > 4 else 'low'} indication of accuracy.",
                "completeness": f"Response thoroughness: {completeness_score:.1f}/10. Response is {'comprehensive' if completeness_score > 7 else 'adequate' if completeness_score > 4 else 'incomplete'}."
            }
            
            eval_time = time.time() - start_time
            
            return {
                "overall_score": round(overall_score, 2),
                "detailed_scores": detailed_scores,
                "detailed_feedback": detailed_feedback,
                "weights_used": weights,
                "evaluation_time": round(eval_time, 2),
                "model_info": {
                    "name": self.model_name,
                    "device": self.device,
                    "evaluation_method": "sentence_transformer_semantic"
                },
                "input_info": {
                    "prompt_length": len(prompt),
                    "response_length": len(response)
                },
                "error": False
            }
            
        except Exception as e:
            eval_time = time.time() - start_time
            logger.error(f"Simple judge evaluation failed: {e}")
            return {
                "overall_score": None,
                "detailed_scores": {aspect: None for aspect in weights.keys()},
                "detailed_feedback": {aspect: f"Evaluation failed: {str(e)}" for aspect in weights.keys()},
                "weights_used": weights,
                "evaluation_time": round(eval_time, 2),
                "model_info": {
                    "name": self.model_name,
                    "device": self.device,
                    "evaluation_method": "sentence_transformer_semantic"
                },
                "error": True,
                "error_message": f"Evaluation failed: {str(e)}"
            }
    
    def compare_responses(self, prompt: str, response1: str, response2: str) -> Dict[str, Any]:
        """Compare two responses"""
        logger.info("Starting simple judge response comparison")
        
        eval1 = self.evaluate(prompt, response1)
        eval2 = self.evaluate(prompt, response2)
        
        if eval1.get("error") or eval2.get("error"):
            return {
                "winner": "error",
                "margin": 0,
                "response1_evaluation": eval1,
                "response2_evaluation": eval2,
                "comparison_summary": {
                    "response1_score": None,
                    "response2_score": None,
                    "difference": 0
                },
                "error": True
            }
        
        score1 = eval1["overall_score"]
        score2 = eval2["overall_score"]
        
        if score1 > score2:
            winner = "response1"
            margin = score1 - score2
        elif score2 > score1:
            winner = "response2"  
            margin = score2 - score1
        else:
            winner = "tie"
            margin = 0
        
        return {
            "winner": winner,
            "margin": round(margin, 2),
            "response1_evaluation": eval1,
            "response2_evaluation": eval2,
            "comparison_summary": {
                "response1_score": score1,
                "response2_score": score2,
                "difference": round(score1 - score2, 2)
            },
            "error": False
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the simple judge model"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "model_type": "sentence_transformer_based",
            "evaluation_method": "semantic_similarity_plus_rules",
            "available_aspects": ["relevance", "coherence", "accuracy", "completeness"],
            "model_description": "Simple but effective judge using sentence transformers for semantic analysis",
            "strengths": ["Fast", "Reliable", "No complex dependencies", "Interpretable"],
            "limitations": ["Rule-based components", "Not fine-tuned for judging"]
        }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.model_loaded = False
        logger.info("Simple judge model unloaded")


# Singleton instance for efficient resource usage
_simple_judge_instance = None

def get_simple_judge() -> SimpleJudge:
    """Get singleton instance of SimpleJudge"""
    global _simple_judge_instance
    if _simple_judge_instance is None:
        _simple_judge_instance = SimpleJudge()
    return _simple_judge_instance
