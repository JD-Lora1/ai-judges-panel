"""
Specialized LLM Judge Implementation
Single, efficient LLM judge using fine-tuned model specifically trained for evaluation tasks.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import logging
from typing import Dict, Any, Optional
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMJudge:
    """
    Single AI judge using specialized LLM model fine-tuned for evaluation tasks.
    Provides accurate and contextual evaluation of responses.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "vwxyzjn/online_dpo_llmjudge"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 200  # Longer for proper judge responses
        self.model_loaded = False
        
        logger.info(f"LLMJudge initialized - Device: {self.device}")
    
    def _load_model(self):
        """Load the specialized LLM judge model and tokenizer (lazy loading)"""
        if self.model_loaded:
            return
            
        try:
            logger.info(f"Loading LLM Judge model from {self.model_name}")
            start_time = time.time()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"LLM Judge model loaded successfully in {load_time:.2f} seconds")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load specialized LLM judge model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _generate_evaluation(self, prompt: str, response: str, aspect: str) -> Dict[str, Any]:
        """Generate evaluation for a specific aspect using specialized LLM judge"""
        if not self.model_loaded:
            try:
                self._load_model()
            except Exception as e:
                logger.error(f"Failed to load model for evaluation: {e}")
                return {
                    "score": None,
                    "feedback": f"LLM evaluation is not working - Model loading failed: {str(e)}",
                    "aspect": aspect,
                    "error": True
                }
        
        # Create a proper evaluation prompt for LLM judge
        evaluation_prompt = f"""Please evaluate the following answer based on its {aspect}.

User Question: {prompt}

Assistant Answer: {response}

Please rate the {aspect} of this answer on a scale from 1 to 10, where:
- 1-3: Poor {aspect}
- 4-6: Average {aspect} 
- 7-8: Good {aspect}
- 9-10: Excellent {aspect}

Provide your rating as a number followed by a brief explanation.

Rating: """

        try:
            # Tokenize input with proper context length
            inputs = self.tokenizer(
                evaluation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,  # Longer context for better evaluation
                padding=True
            ).to(self.device)
            
            # Generate response optimized for judge model
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Enough for score and brief explanation
                    do_sample=True,  # Allow some variation
                    temperature=0.3,  # Low temperature for consistent evaluation
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            logger.info(f"LLM Judge raw response for {aspect}: {generated_text[:100]}...")
            
            # Extract score and feedback
            score, feedback = self._parse_evaluation_response(generated_text, aspect)
            
            if score is None:
                return {
                    "score": None,
                    "feedback": f"LLM evaluation is not working - Could not parse response: {generated_text[:100]}...",
                    "aspect": aspect,
                    "error": True
                }
            
            return {
                "score": score,
                "feedback": feedback,
                "aspect": aspect,
                "error": False,
                "raw_response": generated_text[:200]
            }
            
        except Exception as e:
            logger.error(f"Evaluation generation failed for {aspect}: {e}")
            return {
                "score": None,
                "feedback": f"LLM evaluation is not working - Generation failed: {str(e)}",
                "aspect": aspect,
                "error": True
            }
    
    def _parse_evaluation_response(self, response_text: str, aspect: str) -> tuple:
        """Parse the model's conversational evaluation response"""
        try:
            score = None
            reasoning = ""
            
            # Clean the response
            response_text = response_text.strip()
            
            # Look for any number in the response (simplified approach)
            import re
            
            # First try to find numbers directly at start of response
            first_word = response_text.strip().split()[0] if response_text.strip() else ""
            if first_word.replace('.', '').isdigit():
                try:
                    score = float(first_word)
                    if 1 <= score <= 10:
                        pass  # Valid score
                    elif 10 < score <= 100:  # Percentage
                        score = score / 10
                    else:
                        score = None
                except ValueError:
                    score = None
            else:
                # Look for any number in the text
                numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response_text)
                score = None
                for num_str in numbers:
                    try:
                        potential_score = float(num_str)
                        if 1 <= potential_score <= 10:
                            score = potential_score
                            break
                        elif 10 < potential_score <= 100:  # Percentage
                            score = potential_score / 10
                            break
                    except ValueError:
                        continue
            
            # Final fallback - if still no score, return None to indicate failure
            if score is None:
                logger.warning(f"Could not extract score from response: {response_text[:100]}...")
                return None, f"Failed to parse {aspect} evaluation from: {response_text[:100]}..."
            
            # Create feedback with LLM judge response
            feedback = f"Judge Score: {score}/10 for {aspect}. {response_text[:100] if response_text else 'No detailed feedback'}"
            
            return score, feedback
            
        except Exception as e:
            logger.error(f"Failed to parse evaluation response for {aspect}: {e}")
            return None, f"Parsing failed: {str(e)[:50]}"
    
    def evaluate(self, prompt: str, response: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation using GPT-2 model
        
        Args:
            prompt: Input prompt to evaluate against
            response: Response to evaluate
            weights: Optional weights for different aspects
            
        Returns:
            Dictionary containing scores, feedback, and metrics
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
        evaluations = {}
        failed_aspects = []
        
        # Evaluate each aspect
        for aspect in weights.keys():
            logger.info(f"Evaluating {aspect}...")
            evaluation = self._generate_evaluation(prompt, response, aspect)
            evaluations[aspect] = evaluation
            
            # Track failed evaluations
            if evaluation.get("error", False) or evaluation.get("score") is None:
                failed_aspects.append(aspect)
        
        # If any evaluation failed, return error result
        if failed_aspects:
            eval_time = time.time() - start_time
            error_msg = f"LLM evaluation is not working - Failed aspects: {', '.join(failed_aspects)}"
            
            return {
                "overall_score": None,
                "detailed_scores": {aspect: None for aspect in weights.keys()},
                "detailed_feedback": {aspect: evaluations[aspect]["feedback"] for aspect in weights.keys()},
                "weights_used": weights,
                "evaluation_time": round(eval_time, 2),
                "model_info": {
                    "name": self.model_name,
                    "device": self.device,
                    "aspects_evaluated": list(weights.keys())
                },
                "input_info": {
                    "prompt_length": len(prompt),
                    "response_length": len(response)
                },
                "error": True,
                "error_message": error_msg
            }
        
        # Calculate weighted overall score (only if all evaluations succeeded)
        overall_score = sum(
            evaluations[aspect]["score"] * weights[aspect] 
            for aspect in weights.keys()
            if evaluations[aspect]["score"] is not None
        )
        
        # Calculate evaluation time
        eval_time = time.time() - start_time
        
        # Prepare detailed results
        detailed_scores = {aspect: evaluations[aspect]["score"] for aspect in weights.keys()}
        detailed_feedback = {aspect: evaluations[aspect]["feedback"] for aspect in weights.keys()}
        
        result = {
            "overall_score": round(overall_score, 2),
            "detailed_scores": detailed_scores,
            "detailed_feedback": detailed_feedback,
            "weights_used": weights,
            "evaluation_time": round(eval_time, 2),
            "model_info": {
                "name": self.model_name,
                "device": self.device,
                "aspects_evaluated": list(weights.keys())
            },
            "input_info": {
                "prompt_length": len(prompt),
                "response_length": len(response)
            },
            "error": False,
            "raw_responses": {aspect: evaluations[aspect].get("raw_response", "") for aspect in weights.keys()}
        }
        
        logger.info(f"Evaluation completed in {eval_time:.2f}s - Overall score: {overall_score:.2f}")
        return result
    
    def batch_evaluate(self, evaluations_data: list) -> list:
        """Batch evaluation for multiple prompt-response pairs"""
        results = []
        
        logger.info(f"Starting batch evaluation of {len(evaluations_data)} items")
        
        for i, data in enumerate(evaluations_data):
            logger.info(f"Processing batch item {i+1}/{len(evaluations_data)}")
            
            prompt = data.get("prompt", "")
            response = data.get("response", "")
            weights = data.get("weights")
            
            result = self.evaluate(prompt, response, weights)
            result["batch_index"] = i
            results.append(result)
            
            # Memory cleanup between evaluations
            if i % 5 == 0:  # Every 5 evaluations
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        
        logger.info("Batch evaluation completed")
        return results
    
    def compare_responses(self, prompt: str, response1: str, response2: str) -> Dict[str, Any]:
        """Compare two responses to the same prompt"""
        logger.info("Starting response comparison")
        
        eval1 = self.evaluate(prompt, response1)
        eval2 = self.evaluate(prompt, response2)
        
        # Determine winner
        score1 = eval1.get("overall_score")
        score2 = eval2.get("overall_score")
        
        # Handle None scores
        if score1 is None or score2 is None:
            winner = "error"
            margin = 0
            if score1 is None and score2 is None:
                winner = "both_failed"
            elif score1 is None:
                winner = "response1_failed"
            else:
                winner = "response2_failed"
        elif score1 > score2:
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
            "margin": round(margin, 2) if margin is not None else 0,
            "response1_evaluation": eval1,
            "response2_evaluation": eval2,
            "comparison_summary": {
                "response1_score": score1,
                "response2_score": score2,
                "difference": round(margin, 2) if margin is not None else 0
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "max_length": self.max_length,
            "available_aspects": ["relevance", "coherence", "accuracy", "completeness"],
            "model_parameters": "7B",
            "model_description": "Specialized LLM Judge - Fine-tuned for evaluating response quality and relevance"
        }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        self.model_loaded = False
        logger.info("Model unloaded and memory cleared")


# Singleton instance for efficient resource usage
_llm_judge_instance = None

def get_llm_judge() -> LLMJudge:
    """Get singleton instance of LLMJudge"""
    global _llm_judge_instance
    if _llm_judge_instance is None:
        _llm_judge_instance = LLMJudge()
    return _llm_judge_instance

# Keep backward compatibility
get_phi2_judge = get_llm_judge
get_gpt2_judge = get_llm_judge
Phi2Judge = LLMJudge
GPT2Judge = LLMJudge
