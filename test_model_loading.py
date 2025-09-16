#!/usr/bin/env python3
"""
Simple test to check if the LLM judge model loads correctly
"""

import sys
import time
from app.models.phi2_judge import get_phi2_judge

def test_model_loading():
    """Test if model loads without errors"""
    print("Testing model loading...")
    
    try:
        # Get judge instance
        judge = get_phi2_judge()
        print(f"Judge instance created: {type(judge).__name__}")
        
        # Get model info
        info = judge.get_model_info()
        print(f"Model name: {info['model_name']}")
        print(f"Model loaded: {info['model_loaded']}")
        print(f"Device: {info['device']}")
        print(f"Description: {info['model_description']}")
        
        if not info['model_loaded']:
            print("Attempting to load model...")
            start_time = time.time()
            
            # Try a simple evaluation to trigger model loading
            result = judge.evaluate("What is Python?", "Python is a programming language.", {"relevance": 1.0})
            
            load_time = time.time() - start_time
            print(f"Model loaded successfully in {load_time:.2f} seconds")
            print(f"Test evaluation result: {result.get('overall_score')}")
            
        return True
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
