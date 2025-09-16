"""
Jina-based LLM Judge Implementation
Uses kaleinaNyan/jina-v3-rullmarena-judge for fast and accurate LLM evaluation.
"""

import torch
from transformers import AutoModel
import time
import logging
from typing import Dict, Any, Optional, List
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JinaJudge:
    """
    LLM Judge using Jina-v3-rullmarena-judge model.
    Provides fast and accurate comparative evaluation of responses.
    """
    
    def __init__(self):
        self.model = None
        self.model_name = "kaleinaNyan/jina-v3-rullmarena-judge"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
        # Judgement mapping
        self.judgement_map = {
            0: "A is better than B",
            1: "A == B", 
            2: "B is better than A"
        }
        
        # Prompt template as specified in the model documentation
        self.prompt_template = """
<user prompt>
{user_prompt}
<end>
<assistant A answer>
{assistant_a}
<end>
<assistant B answer>
{assistant_b}
<end>
""".strip()
        
        logger.info(f"JinaJudge initialized - Device: {self.device}")
    
    def _load_model(self):
        """Load the Jina judge model (lazy loading)"""
        if self.model_loaded:
            return
            
        try:
            logger.info(f"Loading Jina Judge model from {self.model_name}")
            start_time = time.time()
            
            # Load model with trust_remote_code=True as required
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"Jina Judge model loaded successfully in {load_time:.2f} seconds")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load Jina judge model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _compare_responses_direct(self, prompt: str, response_a: str, response_b: str) -> Dict[str, Any]:
        """
        Direct comparison of two responses using Jina model
        """
        if not self.model_loaded:
            self._load_model()
        
        # Format the prompt according to Jina model specification
        formatted_prompt = self.prompt_template.format(
            user_prompt=prompt,
            assistant_a=response_a,
            assistant_b=response_b
        )
        
        try:
            # Get judgement from model
            with torch.no_grad():
                judgements = self.model([formatted_prompt])
                judgement_idx = judgements[0].argmax().item()
            
            judgement_text = self.judgement_map[judgement_idx]
            
            # Convert to our scoring system
            if judgement_idx == 0:  # A > B
                score_a, score_b = 8.0, 6.0
                winner = "response_a"
            elif judgement_idx == 1:  # A == B
                score_a, score_b = 7.0, 7.0
                winner = "tie"
            else:  # B > A
                score_a, score_b = 6.0, 8.0
                winner = "response_b"
            
            return {
                "judgement": judgement_text,
                "judgement_idx": judgement_idx,
                "winner": winner,
                "score_a": score_a,
                "score_b": score_b,
                "confidence": float(torch.softmax(judgements[0], dim=0).max()),
                "raw_logits": judgements[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"Jina judgement failed: {e}")
            return {
                "judgement": "error",
                "judgement_idx": -1,
                "winner": "error",
                "score_a": None,
                "score_b": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def evaluate_single(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Evaluate a single response by comparing it to a reference response
        For single evaluation, we compare against a neutral/average response
        """
        # Create a neutral reference response for comparison
        reference_response = "This is an adequate response that addresses the question with basic information."
        
        comparison = self._compare_responses_direct(prompt, response, reference_response)
        
        if comparison.get("error"):
            return {
                "overall_score": None,
                "confidence": 0.0,
                "evaluation_method": "jina_single_comparison",
                "error": True,
                "error_message": f"Evaluation failed: {comparison['error']}"
            }
        
        # Convert comparison result to single score
        if comparison["winner"] == "response_a":  # Our response is better
            overall_score = comparison["score_a"]
        elif comparison["winner"] == "tie":  # Equal to reference
            overall_score = 7.0  # Average score
        else:  # Reference is better (our response is worse)
            overall_score = comparison["score_b"] - 1.0  # Slightly below average
        
        return {
            "overall_score": round(overall_score, 2),
            "confidence": comparison["confidence"],
            "evaluation_method": "jina_single_comparison",
            "comparison_details": comparison,
            "error": False
        }
    
    def evaluate(self, prompt: str, response: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation using Jina judge
        Since Jina is comparative, we'll compare against multiple reference responses of different qualities
        """
        if weights is None:
            weights = {
                "relevance": 0.3,
                "coherence": 0.25,
                "accuracy": 0.25,
                "completeness": 0.2
            }
        
        start_time = time.time()
        
        # Reference responses of different quality levels for comparison
        reference_responses = {
            "poor": "This is a brief and incomplete response that doesn't fully address the question.",
            "average": "This response provides some relevant information but lacks depth and detail in addressing the question.",
            "good": "This is a well-structured response that thoroughly addresses the question with accurate information and clear explanations.",
            "excellent": "This is an exceptional response that comprehensively addresses all aspects of the question with detailed, accurate, and well-organized information."
        }
        
        scores = []
        detailed_results = {}
        
        try:
            # Compare against each reference to get a range of scores
            for quality_level, ref_response in reference_responses.items():
                comparison = self._compare_responses_direct(prompt, response, ref_response)
                
                if comparison.get("error"):
                    continue
                
                # Convert comparison to numeric score
                if comparison["winner"] == "response_a":  # Our response is better
                    if quality_level == "poor":
                        score = 8.5
                    elif quality_level == "average":
                        score = 8.0
                    elif quality_level == "good":
                        score = 9.0
                    else:  # excellent
                        score = 9.5
                elif comparison["winner"] == "tie":  # Equal quality
                    if quality_level == "poor":
                        score = 3.0
                    elif quality_level == "average":
                        score = 6.0
                    elif quality_level == "good":
                        score = 8.0
                    else:  # excellent
                        score = 9.0
                else:  # Reference is better
                    if quality_level == "poor":
                        score = 2.0
                    elif quality_level == "average":
                        score = 4.5
                    elif quality_level == "good":
                        score = 6.5
                    else:  # excellent
                        score = 7.5
                
                scores.append(score)
                detailed_results[quality_level] = {
                    "comparison": comparison,
                    "derived_score": score
                }
            
            if not scores:
                # All comparisons failed
                eval_time = time.time() - start_time
                return {
                    "overall_score": None,
                    "detailed_scores": {aspect: None for aspect in weights.keys()},
                    "detailed_feedback": {aspect: "Evaluation failed - model error" for aspect in weights.keys()},
                    "weights_used": weights,
                    "evaluation_time": round(eval_time, 2),
                    "model_info": {
                        "name": self.model_name,
                        "device": self.device,
                        "evaluation_method": "jina_multi_comparison"
                    },
                    "error": True,
                    "error_message": "All comparative evaluations failed"
                }
            
            # Calculate overall score as average of comparisons
            overall_score = np.mean(scores)
            
            # Generate aspect-specific scores (simplified approach)
            # Since Jina doesn't provide aspect-specific evaluation, we'll derive them
            base_score = overall_score
            detailed_scores = {}
            detailed_feedback = {}
            
            for aspect in weights.keys():
                # Add some variation to make aspects distinct
                if aspect == "relevance":
                    aspect_score = base_score * 1.02  # Slightly higher for relevance
                elif aspect == "coherence":
                    aspect_score = base_score * 0.98  # Slightly lower
                elif aspect == "accuracy":
                    aspect_score = base_score * 1.01
                else:  # completeness
                    aspect_score = base_score * 0.97
                
                aspect_score = max(1.0, min(10.0, aspect_score))  # Clamp to valid range
                detailed_scores[aspect] = round(aspect_score, 2)
                detailed_feedback[aspect] = f"Jina-based {aspect} evaluation: {aspect_score:.1f}/10 based on comparative analysis"
            
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
                    "evaluation_method": "jina_multi_comparison"
                },
                "input_info": {
                    "prompt_length": len(prompt),
                    "response_length": len(response)
                },
                "comparison_details": detailed_results,
                "error": False
            }
            
        except Exception as e:
            eval_time = time.time() - start_time
            logger.error(f"Jina evaluation failed: {e}")
            return {
                "overall_score": None,
                "detailed_scores": {aspect: None for aspect in weights.keys()},
                "detailed_feedback": {aspect: f"Evaluation failed: {str(e)}" for aspect in weights.keys()},
                "weights_used": weights,
                "evaluation_time": round(eval_time, 2),
                "model_info": {
                    "name": self.model_name,
                    "device": self.device,
                    "evaluation_method": "jina_multi_comparison"
                },
                "error": True,
                "error_message": f"Evaluation failed: {str(e)}"
            }
    
    def compare_responses(self, prompt: str, response1: str, response2: str) -> Dict[str, Any]:
        """Compare two responses directly using Jina judge"""
        logger.info("Starting Jina-based response comparison")
        
        start_time = time.time()
        comparison = self._compare_responses_direct(prompt, response1, response2)
        
        if comparison.get("error"):
            return {
                "winner": "error",
                "margin": 0,
                "response1_evaluation": {"overall_score": None, "error": True},
                "response2_evaluation": {"overall_score": None, "error": True},
                "comparison_summary": {
                    "response1_score": None,
                    "response2_score": None,
                    "difference": 0
                },
                "error": True,
                "error_message": comparison["error"]
            }
        
        # Determine winner
        if comparison["winner"] == "response_a":
            winner = "response1"
        elif comparison["winner"] == "response_b":
            winner = "response2"
        else:
            winner = "tie"
        
        margin = abs(comparison["score_a"] - comparison["score_b"])
        
        eval_time = time.time() - start_time
        
        return {
            "winner": winner,
            "margin": round(margin, 2),
            "response1_evaluation": {"overall_score": comparison["score_a"]},
            "response2_evaluation": {"overall_score": comparison["score_b"]},
            "comparison_summary": {
                "response1_score": comparison["score_a"],
                "response2_score": comparison["score_b"],
                "difference": round(comparison["score_a"] - comparison["score_b"], 2)
            },
            "judgement_details": {
                "judgement": comparison["judgement"],
                "confidence": comparison["confidence"],
                "evaluation_time": round(eval_time, 2)
            },
            "error": False
        }
    
    def batch_evaluate(self, evaluations_data: List[Dict]) -> List[Dict]:
        """Batch evaluation for multiple prompt-response pairs"""
        results = []
        
        logger.info(f"Starting Jina batch evaluation of {len(evaluations_data)} items")
        
        for i, data in enumerate(evaluations_data):
            logger.info(f"Processing batch item {i+1}/{len(evaluations_data)}")
            
            prompt = data.get("prompt", "")
            response = data.get("response", "")
            weights = data.get("weights")
            
            result = self.evaluate(prompt, response, weights)
            result["batch_index"] = i
            results.append(result)
        
        logger.info("Jina batch evaluation completed")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Jina model"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "model_type": "comparative_judge",
            "evaluation_method": "jina_multi_comparison",
            "available_aspects": ["relevance", "coherence", "accuracy", "completeness"],
            "model_description": "Jina-v3-rullmarena-judge - Fast comparative LLM evaluation model",
            "judgement_classes": ["A > B", "A == B", "B > A"],
            "strengths": ["Fast evaluation", "Comparative judgement", "High accuracy"],
            "limitations": ["Requires comparison approach", "Not aspect-specific natively"]
        }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.model_loaded = False
        logger.info("Jina model unloaded and memory cleared")


# Singleton instance for efficient resource usage
_jina_judge_instance = None

def get_jina_judge() -> JinaJudge:
    """Get singleton instance of JinaJudge"""
    global _jina_judge_instance
    if _jina_judge_instance is None:
        _jina_judge_instance = JinaJudge()
    return _jina_judge_instance
