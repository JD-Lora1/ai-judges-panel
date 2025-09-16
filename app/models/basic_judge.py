"""
Basic LLM Judge Implementation
Uses simple text analysis without external ML dependencies for reliable evaluation.
"""

import time
import logging
from typing import Dict, Any, Optional, List
import re
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicJudge:
    """
    Basic but effective LLM Judge using text analysis and linguistic heuristics.
    No external ML dependencies - just pure Python analysis.
    """
    
    def __init__(self):
        self.model_name = "basic_text_analyzer"
        self.model_loaded = True  # Always "loaded" since it's rule-based
        
        # Common words that indicate quality
        self.quality_indicators = {
            'specific_words': ['specifically', 'particularly', 'namely', 'including', 'such as', 'for example', 'instance'],
            'explanation_words': ['because', 'since', 'due to', 'as a result', 'therefore', 'consequently', 'thus'],
            'structure_words': ['first', 'second', 'third', 'finally', 'moreover', 'furthermore', 'additionally', 'however'],
            'technical_indicators': ['process', 'method', 'system', 'mechanism', 'function', 'principle', 'concept']
        }
        
        logger.info(f"BasicJudge initialized - Pure Python text analysis")
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate simple word overlap between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove very common words (stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_relevance(self, prompt: str, response: str) -> float:
        """Calculate relevance using word overlap and keyword matching"""
        try:
            # Basic word overlap
            overlap_score = self._calculate_word_overlap(prompt, response)
            
            # Convert to 1-10 scale
            base_score = 1 + (overlap_score * 9)  # Maps 0->1, 1->10
            
            # Bonus for addressing question words
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
            prompt_lower = prompt.lower()
            response_lower = response.lower()
            
            for qword in question_words:
                if qword in prompt_lower:
                    # Check if response seems to address this type of question
                    if qword == 'what' and ('is' in response_lower or 'are' in response_lower):
                        base_score += 0.5
                    elif qword == 'how' and ('by' in response_lower or 'through' in response_lower):
                        base_score += 0.5
                    elif qword == 'why' and ('because' in response_lower or 'due to' in response_lower):
                        base_score += 0.5
            
            return max(1.0, min(10.0, base_score))
            
        except Exception as e:
            logger.error(f"Relevance calculation failed: {e}")
            return 5.0  # Default middle score
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence based on text structure and flow"""
        try:
            score = 5.0  # Base score
            
            # Length considerations
            word_count = len(response.split())
            if 20 <= word_count <= 200:
                score += 1.5
            elif word_count > 10:
                score += 0.5
            elif word_count < 5:
                score -= 2.0
            
            # Sentence structure
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            if len(sentences) >= 2:
                score += 1.0
            
            # Proper capitalization
            if response and response[0].isupper():
                score += 0.5
            
            # Logical flow indicators
            for word_list in self.quality_indicators.values():
                if any(word in response.lower() for word in word_list):
                    score += 0.3
            
            # Penalty for repetition
            words = response.lower().split()
            if len(words) > 0:
                unique_words = len(set(words))
                repetition_ratio = unique_words / len(words)
                if repetition_ratio < 0.5:
                    score -= 1.0
                elif repetition_ratio > 0.8:
                    score += 0.5
            
            return max(1.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {e}")
            return 5.0
    
    def _calculate_accuracy(self, prompt: str, response: str) -> float:
        """Calculate accuracy based on response indicators"""
        try:
            score = 5.0  # Base score
            
            # Detailed responses tend to be more accurate
            word_count = len(response.split())
            if word_count > 50:
                score += 1.5
            elif word_count > 30:
                score += 1.0
            
            # Numbers and facts
            if re.search(r'\d+', response):
                score += 1.0
            
            # Specific examples
            if any(word in response.lower() for word in self.quality_indicators['specific_words']):
                score += 1.0
            
            # Technical terminology (suggests domain knowledge)
            if any(word in response.lower() for word in self.quality_indicators['technical_indicators']):
                score += 0.5
            
            # Explanatory language
            if any(word in response.lower() for word in self.quality_indicators['explanation_words']):
                score += 0.5
            
            # Penalty for vague language
            vague_indicators = ['thing', 'stuff', 'something', 'somehow', 'sort of', 'kind of']
            vague_count = sum(1 for word in vague_indicators if word in response.lower())
            score -= vague_count * 0.5
            
            # Penalty for obvious errors (basic checks)
            if 'i don\'t know' in response.lower() or 'not sure' in response.lower():
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
                score += 2.5
            elif word_count > 50:
                score += 1.5
            elif word_count > 20:
                score += 1.0
            elif word_count < 10:
                score -= 2.0
            elif word_count < 5:
                score -= 3.0
            
            # Structured responses (lists, numbered points)
            if re.search(r'\\d+\\.', response) or re.search(r'\\n[-*•]', response):
                score += 1.5
            
            # Multiple aspects covered
            sentence_count = len([s for s in response.split('.') if s.strip()])
            if sentence_count >= 5:
                score += 1.0
            elif sentence_count >= 3:
                score += 0.5
            
            # Comprehensive coverage indicators
            coverage_words = ['include', 'such as', 'also', 'additionally', 'furthermore', 'moreover']
            coverage_count = sum(1 for word in coverage_words if word in response.lower())
            score += min(coverage_count * 0.3, 1.0)
            
            # Question type considerations
            if 'explain' in prompt.lower() or 'describe' in prompt.lower():
                # Should have explanatory content
                if any(word in response.lower() for word in self.quality_indicators['explanation_words']):
                    score += 0.5
            
            return max(1.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Completeness calculation failed: {e}")
            return 5.0
    
    def evaluate(self, prompt: str, response: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation using basic text analysis
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
            # Handle edge cases
            if not response or not response.strip():
                return {
                    "overall_score": 1.0,
                    "detailed_scores": {aspect: 1.0 for aspect in weights.keys()},
                    "detailed_feedback": {aspect: "Empty response" for aspect in weights.keys()},
                    "weights_used": weights,
                    "evaluation_time": 0.01,
                    "model_info": {"name": self.model_name, "evaluation_method": "basic_text_analysis"},
                    "input_info": {"prompt_length": len(prompt), "response_length": 0},
                    "error": False
                }
            
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
            def get_quality_description(score):
                if score >= 8.5:
                    return "excellent"
                elif score >= 7.0:
                    return "good"
                elif score >= 5.5:
                    return "adequate"
                elif score >= 4.0:
                    return "below average"
                else:
                    return "poor"
            
            detailed_feedback = {
                "relevance": f"Topic relevance: {relevance_score:.1f}/10 - {get_quality_description(relevance_score)}. Word overlap and topic alignment analysis.",
                "coherence": f"Text coherence: {coherence_score:.1f}/10 - {get_quality_description(coherence_score)}. Structure, flow, and logical organization assessment.",
                "accuracy": f"Content accuracy indicators: {accuracy_score:.1f}/10 - {get_quality_description(accuracy_score)}. Based on specificity, examples, and factual markers.",
                "completeness": f"Response completeness: {completeness_score:.1f}/10 - {get_quality_description(completeness_score)}. Thoroughness and comprehensive coverage analysis."
            }
            
            eval_time = time.time() - start_time
            
            return {
                "overall_score": round(overall_score, 2),
                "detailed_scores": detailed_scores,
                "detailed_feedback": detailed_feedback,
                "weights_used": weights,
                "evaluation_time": round(eval_time, 3),
                "model_info": {
                    "name": self.model_name,
                    "evaluation_method": "basic_text_analysis",
                    "features": ["word_overlap", "structure_analysis", "quality_indicators"]
                },
                "input_info": {
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "word_count": len(response.split())
                },
                "error": False
            }
            
        except Exception as e:
            eval_time = time.time() - start_time
            logger.error(f"Basic judge evaluation failed: {e}")
            return {
                "overall_score": None,
                "detailed_scores": {aspect: None for aspect in weights.keys()},
                "detailed_feedback": {aspect: f"Evaluation failed: {str(e)}" for aspect in weights.keys()},
                "weights_used": weights,
                "evaluation_time": round(eval_time, 3),
                "model_info": {"name": self.model_name, "evaluation_method": "basic_text_analysis"},
                "error": True,
                "error_message": f"Evaluation failed: {str(e)}"
            }
    
    def compare_responses(self, prompt: str, response1: str, response2: str) -> Dict[str, Any]:
        """Compare two responses"""
        logger.info("Starting basic judge response comparison")
        
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
        """Get information about the basic judge"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "model_type": "rule_based_text_analyzer",
            "evaluation_method": "linguistic_heuristics",
            "available_aspects": ["relevance", "coherence", "accuracy", "completeness"],
            "model_description": "Basic text analysis judge using linguistic heuristics and quality indicators",
            "features": ["Word overlap analysis", "Structure assessment", "Quality indicators", "No external dependencies"],
            "strengths": ["Very fast", "No dependencies", "Interpretable", "Always available"],
            "limitations": ["Rule-based", "Limited semantic understanding", "Heuristic-based accuracy"]
        }


# Singleton instance for efficient resource usage
_basic_judge_instance = None

def get_basic_judge() -> BasicJudge:
    """Get singleton instance of BasicJudge"""
    global _basic_judge_instance
    if _basic_judge_instance is None:
        _basic_judge_instance = BasicJudge()
    return _basic_judge_instance
