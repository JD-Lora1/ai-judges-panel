"""
API Package
===========

FastAPI routes and endpoints for the AI Judges Panel.
"""

from .evaluation import router as evaluation_router

__all__ = ["evaluation_router"]
