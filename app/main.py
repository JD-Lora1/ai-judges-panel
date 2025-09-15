"""
AI Judges Panel - Web Application
=================================

FastAPI application for deploying the AI Judges Panel system with Hugging Face models.
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
from .models.hf_judges import HuggingFaceJudgesPanel
from .api.evaluation import router as evaluation_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Judges Panel",
    description="Multi-agent LLM evaluation system using Hugging Face models",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include API routes
app.include_router(evaluation_router, prefix="/api/v1", tags=["evaluation"])

# Global judges panel instance
judges_panel = None

@app.on_event("startup")
async def startup_event():
    """Initialize the judges panel on startup"""
    global judges_panel
    logger.info("🏛️ Initializing AI Judges Panel...")
    try:
        judges_panel = HuggingFaceJudgesPanel()
        await judges_panel.initialize()
        logger.info("✅ AI Judges Panel initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize judges panel: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global judges_panel
    if judges_panel:
        logger.info("🧹 Cleaning up judges panel...")
        await judges_panel.cleanup()

# Pydantic models for API
class EvaluationRequest(BaseModel):
    prompt: str
    response: str
    domain: Optional[str] = None
    include_automatic_metrics: bool = True

class EvaluationResponse(BaseModel):
    final_score: float
    individual_scores: Dict[str, float]
    consensus_level: float
    strengths: List[str]
    improvements: List[str]
    evaluation_time: float
    metadata: Dict[str, Any]

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
    global judges_panel
    
    # Basic health status
    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "judges_panel_ready": False,
        "judges_count": 0
    }
    
    try:
        if judges_panel is not None:
            status["judges_panel_ready"] = judges_panel.is_ready()
            status["judges_count"] = len(judges_panel.judges)
            
            # Get judge status
            judges_status = []
            for judge in judges_panel.judges:
                judges_status.append({
                    "name": judge.name,
                    "initialized": judge.is_initialized
                })
            status["judges_status"] = judges_status
    except Exception as e:
        logger.warning(f"Health check warning: {e}")
        status["warning"] = str(e)
    
    # Return 200 even if judges not ready (for Railway health check)
    return status

# API endpoints
@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
async def evaluate_text(request: EvaluationRequest):
    """Evaluate text using the AI Judges Panel"""
    global judges_panel
    
    if not judges_panel or not judges_panel.is_ready():
        raise HTTPException(
            status_code=503, 
            detail="Judges panel not ready. Please try again later."
        )
    
    try:
        # Perform evaluation
        result = await judges_panel.evaluate(
            prompt=request.prompt,
            response=request.response,
            domain=request.domain,
            include_automatic_metrics=request.include_automatic_metrics
        )
        
        return EvaluationResponse(
            final_score=result.final_score,
            individual_scores=result.individual_scores,
            consensus_level=result.consensus_level,
            strengths=result.strengths,
            improvements=result.improvements,
            evaluation_time=result.evaluation_time,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/v1/judges")
async def get_judges_info():
    """Get information about available judges"""
    global judges_panel
    
    if not judges_panel:
        raise HTTPException(status_code=503, detail="Judges panel not ready")
    
    return judges_panel.get_judges_info()

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
