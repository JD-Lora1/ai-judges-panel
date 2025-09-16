"""
AI Judge Models Package
======================

Contains AI judges implementation using GPT-2 model.
"""

from .phi2_judge import GPT2Judge, get_gpt2_judge, get_phi2_judge

__all__ = ["GPT2Judge", "get_gpt2_judge", "get_phi2_judge"]
