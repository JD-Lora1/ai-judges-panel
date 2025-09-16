"""
Phi-2 Based AI Judge Implementation
Single, efficient LLM judge using Microsoft Phi-2 model for evaluation tasks.
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

class Phi2Judge:
    """
    Single AI judge using Microsoft Phi-2 model for comprehensive evaluation.
    Optimized for resource efficiency and performance.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "microsoft/phi-2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 100  # Short responses for efficiency
        self.model_loaded = False
        
        logger.info(f"Phi2Judge initialized - Device: {self.device}")
    
    def _load_model(self):
        """Load the Phi-2 model and tokenizer (lazy loading)"""
        if self.model_loaded:
            return
            
        try:
            logger.info(f"Loading Phi-2 model from {self.model_name}")
            start_time = time.time()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
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
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load Phi-2 model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _generate_evaluation(self, prompt: str, response: str, aspect: str) -> Dict[str, Any]:
        """Generate evaluation for a specific aspect using Phi-2"""
        if not self.model_loaded:
            self._load_model()
        
        # Create evaluation prompt
        evaluation_prompt = f"""Evaluate the following response to a prompt on {aspect}. Provide a score from 1-10 and brief reasoning.

Prompt: {prompt[:200]}...
Response: {response[:200]}...

Evaluation for {aspect.upper()}:
Score (1-10):"""

        try:
            # Tokenize input
            inputs = self.tokenizer(
                evaluation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    do_sample=False,  # Deterministic
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Extract score and feedback
            score, feedback = self._parse_evaluation_response(generated_text)
            
            return {
                "score": score,
                "feedback": feedback,
                "aspect": aspect
            }
            
        except Exception as e:
            logger.error(f"Evaluation generation failed: {e}")
            return {
                "score": 5.0,  # Default score
                "feedback": f"Evaluation failed due to technical error: {str(e)[:100]}",
                "aspect": aspect
            }
    
    def _parse_evaluation_response(self, response_text: str) -> tuple:
        """Parse the model's response to extract score and feedback"""
        try:
            lines = response_text.split('\n')
            score = 5.0  # default
            feedback = "Generated evaluation response"
            
            # Look for score in various formats
            for line in lines:
                line = line.strip().lower()
                if any(keyword in line for keyword in ['score:', 'rating:', 'grade:']):
                    # Extract numeric score
                    words = line.split()
                    for word in words:
                        try:
                            # Look for number patterns like "8", "8/10", "8.5"
                            if '/' in word:
                                score = float(word.split('/')[0])
                            elif word.replace('.', '').isdigit():
                                score = float(word)
                                if score > 10:
                                    score = score / 10  # Handle percentage
                                break
                        except ValueError:
                            continue
                    break
            
            # Clean feedback
            feedback = response_text[:200].strip()
            if not feedback:
                feedback = f"Score: {score}/10"
            
            # Ensure score is in valid range
            score = max(1.0, min(10.0, score))
            
            return score, feedback
            
        except Exception as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
            return 5.0, f"Parsing error: {str(e)[:50]}"
    
    def evaluate(self, prompt: str, response: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation using Phi-2 model
        
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
        
        # Evaluate each aspect
        for aspect in weights.keys():
            logger.info(f"Evaluating {aspect}...")
            evaluation = self._generate_evaluation(prompt, response, aspect)
            evaluations[aspect] = evaluation
        
        # Calculate weighted overall score
        overall_score = sum(
            evaluations[aspect]["score"] * weights[aspect] 
            for aspect in weights.keys()
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
            }
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
                "difference": round(margin, 2)
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
            "model_parameters": "2.7B",
            "model_description": "Microsoft Phi-2 - Efficient transformer model for text generation and evaluation"
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
_phi2_judge_instance = None

def get_phi2_judge() -> Phi2Judge:
    """Get singleton instance of Phi2Judge"""
    global _phi2_judge_instance
    if _phi2_judge_instance is None:
        _phi2_judge_instance = Phi2Judge()
    return _phi2_judge_instance
