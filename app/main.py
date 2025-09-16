"""
AI Judges Panel - Web Application
=================================

FastAPI application for deploying the AI Judges Panel system with Microsoft Phi-2 model.
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
    title="AI Judges Panel - Phi-2",
    description="AI evaluation system powered by Microsoft Phi-2 model",
    version="2.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include API routes
app.include_router(evaluation_router, prefix="/api/v1", tags=["evaluation"])

# Global Phi-2 judge instance
phi2_judge = None

@app.on_event("startup")
async def startup_event():
    """Initialize the Phi-2 judge on startup"""
    global phi2_judge
    logger.info("🏛️ Initializing Phi-2 AI Judge...")
    try:
        phi2_judge = get_phi2_judge()
        logger.info("✅ Phi-2 AI Judge initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Phi-2 judge: {e}")
        # Don't raise here - model will be loaded on first use
        pass

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global phi2_judge
    if phi2_judge:
        logger.info("🧹 Cleaning up Phi-2 judge...")
        phi2_judge.unload_model()

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
    global phi2_judge
    
    # Basic health status
    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "model_ready": False,
        "model_name": "microsoft/phi-2"
    }
    
    try:
        if phi2_judge is not None:
            model_info = phi2_judge.get_model_info()
            status["model_ready"] = model_info["model_loaded"]
            status["device"] = model_info["device"]
            status["available_aspects"] = model_info["available_aspects"]
    except Exception as e:
        logger.warning(f"Health check warning: {e}")
        status["warning"] = str(e)
    
    # Return 200 even if model not loaded (for Railway health check)
    return status

# API endpoints (legacy support - redirects to new API)
@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
async def evaluate_text(request: EvaluationRequest):
    """Evaluate text using Phi-2 model (legacy endpoint)"""
    global phi2_judge
    
    if not phi2_judge:
        phi2_judge = get_phi2_judge()
    
    try:
        # Perform evaluation
        result = phi2_judge.evaluate(
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
    """Get information about the Phi-2 model"""
    global phi2_judge
    
    if not phi2_judge:
        phi2_judge = get_phi2_judge()
    
    return phi2_judge.get_model_info()

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
