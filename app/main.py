"""
AI Judges Panel - Web Application
=================================

FastAPI application for deploying the AI Judges Panel system with OpenAI GPT-2 model.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import time
from typing import Dict, Any, Optional, List
import asyncio
import logging

# Import our core components
from .models.phi2_judge import get_phi2_judge
from .api.evaluation import router as evaluation_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Judges Panel - GPT-2",
    description="AI evaluation system powered by OpenAI GPT-2 model",
    version="2.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include API routes
app.include_router(evaluation_router, prefix="/api/v1", tags=["evaluation"])

# Global GPT-2 judge instance
gpt2_judge = None

@app.on_event("startup")
async def startup_event():
    """Initialize the GPT-2 judge on startup"""
    global gpt2_judge
    logger.info("🏛️ Initializing GPT-2 AI Judge...")
    try:
        # Don't load model immediately - use lazy loading
        from .models.phi2_judge import get_phi2_judge
        gpt2_judge = get_phi2_judge()  # Uses backward compatibility alias
        logger.info("✅ GPT-2 AI Judge initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize GPT-2 judge: {e}")
        # Don't raise here - model will be loaded on first use
        pass

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global gpt2_judge
    if gpt2_judge:
        logger.info("🧹 Cleaning up GPT-2 judge...")
        gpt2_judge.unload_model()

# Pydantic models for API
class EvaluationRequest(BaseModel):
    prompt: str
    response: str
    custom_weights: Optional[Dict[str, float]] = None

class EvaluationResponse(BaseModel):
    overall_score: float
    detailed_scores: Dict[str, float]
    detailed_feedback: Dict[str, str]
    weights_used: Dict[str, float]
    evaluation_time: float
    model_info: Dict[str, Any]
    input_info: Dict[str, Any]

# Web routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/evaluate", response_class=HTMLResponse)
async def evaluate_page(request: Request):
    """Evaluation page"""
    return templates.TemplateResponse("evaluate.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About page"""
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    global gpt2_judge
    
    # Basic health status - always return healthy for deployment
    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "model_ready": False,
        "model_name": "openai-community/gpt2",
        "service": "AI Judges Panel"
    }
    
    try:
        if gpt2_judge is not None:
            model_info = gpt2_judge.get_model_info()
            status["model_ready"] = model_info["model_loaded"]
            status["device"] = model_info["device"]
            status["available_aspects"] = model_info["available_aspects"]
        else:
            # Service is healthy even without model loaded (lazy loading)
            status["model_ready"] = False
            status["note"] = "Model will be loaded on first evaluation request"
    except Exception as e:
        logger.warning(f"Health check warning: {e}")
        status["warning"] = str(e)
        # Still return healthy status for deployment
    
    return status

# API endpoints (legacy support - redirects to new API)
@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
async def evaluate_text(request: EvaluationRequest):
    """Evaluate text using GPT-2 model (legacy endpoint)"""
    global gpt2_judge
    
    if not gpt2_judge:
        from .models.phi2_judge import get_phi2_judge
        gpt2_judge = get_phi2_judge()
    
    try:
        # Perform evaluation
        result = gpt2_judge.evaluate(
            prompt=request.prompt,
            response=request.response,
            weights=request.custom_weights
        )
        
        return EvaluationResponse(
            overall_score=result["overall_score"],
            detailed_scores=result["detailed_scores"],
            detailed_feedback=result["detailed_feedback"],
            weights_used=result["weights_used"],
            evaluation_time=result["evaluation_time"],
            model_info=result["model_info"],
            input_info=result["input_info"]
        )
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/v1/model")
async def get_model_info():
    """Get information about the GPT-2 model"""
    global gpt2_judge
    
    if not gpt2_judge:
        from .models.phi2_judge import get_phi2_judge
        gpt2_judge = get_phi2_judge()
    
    return gpt2_judge.get_model_info()

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error_code": 404,
        "error_message": "Page not found"
    }, status_code=404)

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error_code": 500,
        "error_message": "Internal server error"
    }, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=port,
        reload=True if os.environ.get("ENVIRONMENT") == "development" else False
    )
