"""
API Routes for Evaluation
=========================

FastAPI routes for handling evaluation requests using Phi-2 model.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uuid
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for evaluation results (in production, use Redis or database)
evaluation_cache = {}

class EvaluationRequest(BaseModel):
    """Evaluation request using Phi-2 model"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="The prompt to evaluate against")
    response: str = Field(..., min_length=1, max_length=5000, description="The response to evaluate")
    custom_weights: Optional[Dict[str, float]] = Field(None, description="Custom weights for evaluation aspects")
    evaluation_id: Optional[str] = Field(None, description="Custom evaluation ID")

class EvaluationStatus(BaseModel):
    """Status of an ongoing evaluation"""
    evaluation_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BatchEvaluationRequest(BaseModel):
    """Request for batch evaluation"""
    evaluations: List[EvaluationRequest]
    batch_name: Optional[str] = Field(None, description="Name for this batch")

class ComparisonRequest(BaseModel):
    """Request to compare multiple responses"""
    prompt: str
    responses: Dict[str, str]  # name: response
    custom_weights: Optional[Dict[str, float]] = Field(None, description="Custom weights for comparison")

@router.get("/status")
async def get_api_status():
    """Get API status and statistics"""
    return {
        "status": "active",
        "version": "1.0.0",
        "cached_evaluations": len(evaluation_cache),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/models/info")
async def get_model_info():
    """Get information about the Phi-2 model"""
    try:
        from ..models.phi2_judge import get_phi2_judge
        
        judge = get_phi2_judge()
        model_info = judge.get_model_info()
        
        return {
            "model_info": model_info,
            "status": "active",
            "supported_aspects": model_info["available_aspects"],
            "default_weights": {
                "relevance": 0.3,
                "coherence": 0.25,
                "accuracy": 0.25,
                "completeness": 0.2
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_with_phi2(request: EvaluationRequest):
    """
    Perform evaluation using Phi-2 model
    """
    try:
        # Generate evaluation ID
        eval_id = request.evaluation_id or str(uuid.uuid4())
        
        # Store initial status
        evaluation_cache[eval_id] = EvaluationStatus(
            evaluation_id=eval_id,
            status="processing",
            progress=0.0,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Get Phi-2 judge
        from ..models.phi2_judge import get_phi2_judge
        judge = get_phi2_judge()
        
        # Perform evaluation
        result = judge.evaluate(
            prompt=request.prompt,
            response=request.response,
            weights=request.custom_weights
        )
        
        # Convert to API response format
        result_dict = {
            "evaluation_id": eval_id,
            "overall_score": result["overall_score"],
            "detailed_scores": result["detailed_scores"],
            "detailed_feedback": result["detailed_feedback"],
            "weights_used": result["weights_used"],
            "evaluation_time": result["evaluation_time"],
            "model_info": result["model_info"],
            "input_info": result["input_info"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Update cache
        evaluation_cache[eval_id].status = "completed"
        evaluation_cache[eval_id].progress = 1.0
        evaluation_cache[eval_id].result = result_dict
        evaluation_cache[eval_id].updated_at = datetime.now()
        
        return result_dict
            
    except Exception as e:
        logger.error(f"Phi-2 evaluation failed: {e}")
        
        # Update cache with error
        if 'eval_id' in locals() and eval_id in evaluation_cache:
            evaluation_cache[eval_id].status = "failed"
            evaluation_cache[eval_id].error = str(e)
            evaluation_cache[eval_id].updated_at = datetime.now()
        
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.get("/evaluate/{evaluation_id}/status", response_model=EvaluationStatus)
async def get_evaluation_status(evaluation_id: str):
    """Get the status of a specific evaluation"""
    if evaluation_id not in evaluation_cache:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return evaluation_cache[evaluation_id]

@router.post("/evaluate/batch")
async def batch_evaluate(request: BatchEvaluationRequest):
    """
    Perform batch evaluation using Phi-2 model
    """
    try:
        batch_id = str(uuid.uuid4())
        
        # Get Phi-2 judge
        from ..models.phi2_judge import get_phi2_judge
        judge = get_phi2_judge()
        
        # Prepare batch data
        batch_data = []
        for eval_request in request.evaluations:
            batch_data.append({
                "prompt": eval_request.prompt,
                "response": eval_request.response,
                "weights": eval_request.custom_weights
            })
        
        # Perform batch evaluation
        batch_results = judge.batch_evaluate(batch_data)
        
        # Format results
        results = []
        for i, result in enumerate(batch_results):
            eval_request = request.evaluations[i]
            results.append({
                "index": i,
                "evaluation_id": eval_request.evaluation_id or f"{batch_id}_{i}",
                "overall_score": result["overall_score"],
                "detailed_scores": result["detailed_scores"],
                "evaluation_time": result["evaluation_time"],
                "status": "completed"
            })
        
        return {
            "batch_id": batch_id,
            "batch_name": request.batch_name,
            "total_evaluations": len(request.evaluations),
            "successful_evaluations": len(results),
            "failed_evaluations": 0,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")

@router.post("/evaluate/compare")
async def compare_responses(request: ComparisonRequest):
    """
    Compare multiple responses using Phi-2 model
    """
    try:
        # Get Phi-2 judge
        from ..models.phi2_judge import get_phi2_judge
        judge = get_phi2_judge()
        
        # Handle pair comparison if exactly 2 responses
        if len(request.responses) == 2:
            response_names = list(request.responses.keys())
            response1 = request.responses[response_names[0]]
            response2 = request.responses[response_names[1]]
            
            comparison_result = judge.compare_responses(
                prompt=request.prompt,
                response1=response1,
                response2=response2
            )
            
            return {
                "prompt": request.prompt,
                "comparison_type": "pairwise",
                "winner": comparison_result["winner"],
                "margin": comparison_result["margin"],
                "response_evaluations": {
                    response_names[0]: comparison_result["response1_evaluation"],
                    response_names[1]: comparison_result["response2_evaluation"]
                },
                "comparison_summary": comparison_result["comparison_summary"],
                "timestamp": datetime.now().isoformat()
            }
        
        # Handle multiple responses comparison
        else:
            results = {}
            scores = []
            
            # Evaluate each response
            for name, response in request.responses.items():
                try:
                    result = judge.evaluate(
                        prompt=request.prompt,
                        response=response,
                        weights=request.custom_weights
                    )
                    
                    results[name] = {
                        "overall_score": result["overall_score"],
                        "detailed_scores": result["detailed_scores"],
                        "detailed_feedback": result["detailed_feedback"],
                        "evaluation_time": result["evaluation_time"]
                    }
                    
                    scores.append((name, result["overall_score"]))
                    
                except Exception as e:
                    logger.error(f"Comparison evaluation for '{name}' failed: {e}")
                    results[name] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Generate comparison analysis
            successful_results = [(k, v) for k, v in results.items() if "overall_score" in v]
            
            if successful_results:
                # Sort by score
                scores.sort(key=lambda x: x[1], reverse=True)
                
                best_response = scores[0][0]
                best_score = scores[0][1]
                
                # Calculate aspect averages
                aspect_totals = {}
                aspect_counts = {}
                
                for name, result in successful_results:
                    if "detailed_scores" in result:
                        for aspect, score in result["detailed_scores"].items():
                            if aspect not in aspect_totals:
                                aspect_totals[aspect] = 0
                                aspect_counts[aspect] = 0
                            aspect_totals[aspect] += score
                            aspect_counts[aspect] += 1
                
                aspect_averages = {
                    aspect: total / aspect_counts[aspect]
                    for aspect, total in aspect_totals.items()
                    if aspect_counts[aspect] > 0
                }
                
                comparison_analysis = {
                    "best_response": best_response,
                    "best_score": best_score,
                    "ranking": scores,
                    "aspect_averages": aspect_averages,
                    "total_responses": len(request.responses),
                    "successful_evaluations": len(successful_results),
                    "failed_evaluations": len(request.responses) - len(successful_results)
                }
            else:
                comparison_analysis = {
                    "error": "No successful evaluations to compare",
                    "total_responses": len(request.responses),
                    "successful_evaluations": 0,
                    "failed_evaluations": len(request.responses)
                }
            
            return {
                "prompt": request.prompt,
                "comparison_type": "multiple",
                "results": results,
                "comparison_analysis": comparison_analysis,
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Response comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.get("/evaluate/history")
async def get_evaluation_history(limit: int = 50, offset: int = 0):
    """Get evaluation history with pagination"""
    try:
        # Get all cached evaluations
        all_evaluations = list(evaluation_cache.values())
        
        # Sort by creation time (newest first)
        all_evaluations.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        paginated_evaluations = all_evaluations[offset:offset + limit]
        
        # Convert to response format
        history_items = []
        for eval_status in paginated_evaluations:
            item = {
                "evaluation_id": eval_status.evaluation_id,
                "status": eval_status.status,
                "created_at": eval_status.created_at.isoformat(),
                "updated_at": eval_status.updated_at.isoformat()
            }
            
            if eval_status.result:
                item["final_score"] = eval_status.result.get("final_score")
                item["evaluation_time"] = eval_status.result.get("evaluation_time")
            
            if eval_status.error:
                item["error"] = eval_status.error
            
            history_items.append(item)
        
        return {
            "total": len(all_evaluations),
            "limit": limit,
            "offset": offset,
            "evaluations": history_items
        }
        
    except Exception as e:
        logger.error(f"Failed to get evaluation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.delete("/evaluate/{evaluation_id}")
async def delete_evaluation(evaluation_id: str):
    """Delete a specific evaluation from cache"""
    if evaluation_id not in evaluation_cache:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    del evaluation_cache[evaluation_id]
    return {"message": f"Evaluation {evaluation_id} deleted successfully"}

@router.delete("/evaluate/cache/clear")
async def clear_evaluation_cache():
    """Clear all cached evaluations"""
    cleared_count = len(evaluation_cache)
    evaluation_cache.clear()
    return {"message": f"Cleared {cleared_count} cached evaluations"}

@router.get("/model/default-weights")
async def get_default_weights():
    """Get default evaluation weights for Phi-2 model"""
    try:
        default_weights = {
            "relevance": 0.3,
            "coherence": 0.25,
            "accuracy": 0.25,
            "completeness": 0.2
        }
        
        return {
            "default_weights": default_weights,
            "total_weight": sum(default_weights.values()),
            "supported_aspects": list(default_weights.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get default weights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get default weights: {str(e)}")

@router.post("/model/validate-weights")
async def validate_weights(weights: Dict[str, float]):
    """Validate custom weights for evaluation"""
    try:
        # Check that all weights are valid numbers between 0 and 1
        if not all(0 <= weight <= 1 for weight in weights.values()):
            return {
                "valid": False,
                "error": "All weights must be between 0 and 1",
                "timestamp": datetime.now().isoformat()
            }
        
        # Check that weights sum to a reasonable total (allow some flexibility)
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {
                "valid": False,
                "error": "Weights cannot all be zero",
                "timestamp": datetime.now().isoformat()
            }
        
        # Normalize weights
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        return {
            "valid": True,
            "original_weights": weights,
            "normalized_weights": normalized_weights,
            "total_weight": total_weight,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to validate weights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate weights: {str(e)}")
