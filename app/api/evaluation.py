"""
API Routes for Evaluation
=========================

FastAPI routes for handling evaluation requests and responses.
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

class DetailedEvaluationRequest(BaseModel):
    """Extended evaluation request with additional options"""
    prompt: str = Field(..., min_length=10, max_length=2000, description="The prompt to evaluate against")
    response: str = Field(..., min_length=10, max_length=5000, description="The response to evaluate")
    domain: Optional[str] = Field(None, description="Domain for specialized evaluation")
    custom_weights: Optional[Dict[str, float]] = Field(None, description="Custom weights for judges")
    include_automatic_metrics: bool = Field(True, description="Include automatic metrics")
    evaluation_id: Optional[str] = Field(None, description="Custom evaluation ID")
    model_type: str = Field("hf", description="Model type: 'hf' for HuggingFace judges, 'llm' for LLM judges")
    llm_model: Optional[str] = Field("distilgpt2", description="Specific LLM model for evaluation")

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
    evaluations: List[DetailedEvaluationRequest]
    batch_name: Optional[str] = Field(None, description="Name for this batch")
    model_type: str = Field("hf", description="Default model type for batch")
    llm_model: Optional[str] = Field("distilgpt2", description="Default LLM model for batch")

class ComparisonRequest(BaseModel):
    """Request to compare multiple responses"""
    prompt: str
    responses: Dict[str, str]  # name: response
    domain: Optional[str] = None
    model_type: str = Field("hf", description="Model type for evaluation")
    llm_model: Optional[str] = Field("distilgpt2", description="LLM model for evaluation")

@router.get("/status")
async def get_api_status():
    """Get API status and statistics"""
    return {
        "status": "active",
        "version": "1.0.0",
        "cached_evaluations": len(evaluation_cache),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/models/available")
async def get_available_models():
    """Get list of available LLM models for evaluation"""
    try:
        from ..models.llm_judges import LLMJudgesPanel
        
        models = LLMJudgesPanel.get_available_models()
        model_list = []
        
        for model_id, config in models.items():
            model_list.append({
                "id": model_id,
                "name": config.name,
                "description": config.description,
                "max_length": config.max_length,
                "temperature": config.temperature
            })
        
        return {
            "available_models": model_list,
            "default_model": "distilgpt2",
            "model_types": {
                "hf": "HuggingFace Judges (embeddings + heuristics)",
                "llm": "LLM Judges (language model evaluation)"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@router.post("/evaluate/detailed", response_model=Dict[str, Any])
async def detailed_evaluate(request: DetailedEvaluationRequest, background_tasks: BackgroundTasks):
    """
    Perform detailed evaluation with custom options
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
        
        # Choose appropriate judges panel based on model_type
        if request.model_type == "llm":
            # Use LLM judges
            from ..models.llm_judges import LLMJudgesPanel
            
            llm_panel = LLMJudgesPanel(model_name=request.llm_model)
            
            # Use custom weights if provided, otherwise use defaults
            weights = request.custom_weights or {
                "precision": 0.35,
                "coherence": 0.30,
                "relevance": 0.20,
                "efficiency": 0.10,
                "creativity": 0.05
            }
            
            # Perform LLM-based evaluation
            llm_result = await llm_panel.evaluate_response_async(
                prompt=request.prompt,
                response=request.response,
                weights=weights
            )
            
            # Convert LLM result to standard format
            result_dict = {
                "evaluation_id": eval_id,
                "final_score": llm_result["overall_score"],
                "individual_scores": {
                    "precision": llm_result["individual_scores"]["precision"]["score"],
                    "coherence": llm_result["individual_scores"]["coherence"]["score"],
                    "relevance": llm_result["individual_scores"]["relevance"]["score"],
                    "efficiency": llm_result["individual_scores"]["efficiency"]["score"],
                    "creativity": llm_result["individual_scores"]["creativity"]["score"]
                },
                "consensus_level": llm_result["consensus"],
                "strengths": [
                    llm_result["individual_scores"]["precision"]["feedback"],
                    llm_result["individual_scores"]["coherence"]["feedback"],
                    llm_result["individual_scores"]["relevance"]["feedback"]
                ],
                "improvements": [
                    llm_result["individual_scores"]["efficiency"]["feedback"],
                    llm_result["individual_scores"]["creativity"]["feedback"]
                ],
                "evaluation_time": 0.0,  # Will be calculated
                "metadata": {
                    "model_used": llm_result["model_used"],
                    "model_type": "llm",
                    "weights": llm_result["weights"],
                    "evaluation_stats": llm_result["evaluation_stats"]
                },
                "timestamp": datetime.now().isoformat()
            }
            
        else:
            # Use original HuggingFace judges
            from app.main import judges_panel
            
            if not judges_panel or not judges_panel.is_ready():
                raise HTTPException(status_code=503, detail="Judges panel not ready")
            
            # Update weights if provided
            original_weights = judges_panel.weights.copy()
            if request.custom_weights:
                judges_panel.weights.update(request.custom_weights)
            
            try:
                # Perform evaluation
                result = await judges_panel.evaluate(
                    prompt=request.prompt,
                    response=request.response,
                    domain=request.domain,
                    include_automatic_metrics=request.include_automatic_metrics
                )
            
                # Convert to dict for JSON response
                result_dict = {
                    "evaluation_id": eval_id,
                    "final_score": result.final_score,
                    "individual_scores": result.individual_scores,
                    "consensus_level": result.consensus_level,
                    "strengths": result.strengths,
                    "improvements": result.improvements,
                    "evaluation_time": result.evaluation_time,
                    "metadata": {**result.metadata, "model_type": "hf"},
                    "timestamp": datetime.now().isoformat()
                }
            
            finally:
                # Restore original weights for HF judges
                if request.model_type == "hf":
                    judges_panel.weights = original_weights
            
        # Update cache
        evaluation_cache[eval_id].status = "completed"
        evaluation_cache[eval_id].progress = 1.0
        evaluation_cache[eval_id].result = result_dict
        evaluation_cache[eval_id].updated_at = datetime.now()
        
        return result_dict
            
    except Exception as e:
        logger.error(f"Detailed evaluation failed: {e}")
        
        # Update cache with error
        if eval_id in evaluation_cache:
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
    Perform batch evaluation of multiple texts
    """
    try:
        batch_id = str(uuid.uuid4())
        results = []
        
        from app.main import judges_panel
        
        if not judges_panel or not judges_panel.is_ready():
            raise HTTPException(status_code=503, detail="Judges panel not ready")
        
        # Process each evaluation in the batch
        for i, eval_request in enumerate(request.evaluations):
            try:
                result = await judges_panel.evaluate(
                    prompt=eval_request.prompt,
                    response=eval_request.response,
                    domain=eval_request.domain,
                    include_automatic_metrics=eval_request.include_automatic_metrics
                )
                
                results.append({
                    "index": i,
                    "evaluation_id": eval_request.evaluation_id or f"{batch_id}_{i}",
                    "final_score": result.final_score,
                    "individual_scores": result.individual_scores,
                    "consensus_level": result.consensus_level,
                    "evaluation_time": result.evaluation_time,
                    "status": "completed"
                })
                
            except Exception as e:
                logger.error(f"Batch evaluation item {i} failed: {e}")
                results.append({
                    "index": i,
                    "evaluation_id": eval_request.evaluation_id or f"{batch_id}_{i}",
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "batch_id": batch_id,
            "batch_name": request.batch_name,
            "total_evaluations": len(request.evaluations),
            "successful_evaluations": len([r for r in results if r["status"] == "completed"]),
            "failed_evaluations": len([r for r in results if r["status"] == "failed"]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")

@router.post("/evaluate/compare")
async def compare_responses(request: ComparisonRequest):
    """
    Compare multiple responses to the same prompt
    """
    try:
        from app.main import judges_panel
        
        if not judges_panel or not judges_panel.is_ready():
            raise HTTPException(status_code=503, detail="Judges panel not ready")
        
        results = {}
        comparison_data = []
        
        # Evaluate each response
        for name, response in request.responses.items():
            try:
                result = await judges_panel.evaluate(
                    prompt=request.prompt,
                    response=response,
                    domain=request.domain,
                    include_automatic_metrics=True
                )
                
                results[name] = {
                    "final_score": result.final_score,
                    "individual_scores": result.individual_scores,
                    "consensus_level": result.consensus_level,
                    "strengths": result.strengths[:5],  # Top 5 strengths
                    "improvements": result.improvements[:5],  # Top 5 improvements
                    "evaluation_time": result.evaluation_time
                }
                
                # Prepare data for comparison analysis
                comparison_data.append({
                    "name": name,
                    "final_score": result.final_score,
                    **result.individual_scores
                })
                
            except Exception as e:
                logger.error(f"Comparison evaluation for '{name}' failed: {e}")
                results[name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Generate comparison insights
        successful_results = [r for r in results.values() if "final_score" in r]
        
        if successful_results:
            best_response = max(request.responses.keys(), 
                              key=lambda k: results[k].get("final_score", 0))
            
            # Calculate average scores by aspect
            aspect_averages = {}
            for aspect in ["precision", "coherence", "relevance", "efficiency", "creativity"]:
                scores = [r["individual_scores"].get(aspect, 0) for r in successful_results if "individual_scores" in r]
                if scores:
                    aspect_averages[aspect] = sum(scores) / len(scores)
            
            comparison_analysis = {
                "best_response": best_response,
                "best_score": results[best_response].get("final_score", 0),
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
            "domain": request.domain,
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

@router.get("/judges/weights")
async def get_current_weights():
    """Get current judge weights"""
    try:
        from app.main import judges_panel
        
        if not judges_panel:
            raise HTTPException(status_code=503, detail="Judges panel not ready")
        
        return {
            "weights": judges_panel.weights,
            "total_weight": sum(judges_panel.weights.values()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get weights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get weights: {str(e)}")

@router.post("/judges/weights")
async def update_weights(new_weights: Dict[str, float]):
    """Update judge weights"""
    try:
        from app.main import judges_panel
        
        if not judges_panel:
            raise HTTPException(status_code=503, detail="Judges panel not ready")
        
        # Validate weights
        if not all(0 <= weight <= 1 for weight in new_weights.values()):
            raise HTTPException(status_code=400, detail="All weights must be between 0 and 1")
        
        # Update weights
        judges_panel.weights.update(new_weights)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(judges_panel.weights.values())
        if total_weight > 0:
            judges_panel.weights = {k: v/total_weight for k, v in judges_panel.weights.items()}
        
        return {
            "message": "Weights updated successfully",
            "weights": judges_panel.weights,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update weights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update weights: {str(e)}")
