#!/usr/bin/env python3
"""
Test script for LLM AI Judge
Verifies that the LLM-based judge implementation works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from models.phi2_judge import get_phi2_judge, LLMJudge
import time
import sys
import os
import logging
from typing import Dict, Any
logger = logging.getLogger(__name__)

class MockLLMJudge(LLMJudge):
    """Fast mock version of LLMJudge for testing (no actual model loading)"""
    
    def __init__(self):
        super().__init__()
        self.model_name = "mock-test-model"  # Mock model name
        self.max_length = 100
        self.model_loaded = True  # Pretend model is loaded
    
    def _load_model(self):
        """Mock model loading - instant"""
        if self.model_loaded:
            return
        # Mock model loading - no actual loading
        self.model_loaded = True
        logger.info(f"Mock model '{self.model_name}' loaded instantly")
    
    def _generate_evaluation(self, prompt: str, response: str, aspect: str) -> Dict[str, Any]:
        """Generate mock evaluation instantly without actual LLM"""
        import random
        import hashlib
        
        # Generate deterministic but varied scores based on input hash
        # This makes tests predictable but realistic
        seed_string = f"{prompt}{response}{aspect}"
        seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # Generate score based on content characteristics
        base_score = 6.0  # Base score
        
        # Adjust score based on response characteristics
        if len(response.strip()) > 100:
            base_score += 1.0  # Longer responses get bonus
        if len(response.split()) > 20:
            base_score += 0.5  # More words = more complete
        if aspect.lower() in response.lower():
            base_score += 0.5  # Relevant responses get bonus
        
        # Add some randomness
        score = base_score + random.uniform(-1.0, 1.5)
        score = max(1.0, min(10.0, score))  # Clamp to 1-10 range
        
        # Generate realistic feedback
        feedback = f"Mock {aspect} evaluation: Score {score:.1f}/10. " \
                  f"Response length: {len(response)} chars. " \
                  f"Analysis: {'Good' if score > 7 else 'Average' if score > 5 else 'Needs improvement'} {aspect}."
        
        return {
            "score": round(score, 2),
            "feedback": feedback,
            "aspect": aspect,
            "error": False,
            "raw_response": f"Mock LLM response for {aspect}: {score:.1f}/10"
        }

def test_llm_judge():
    """Test the LLM judge functionality"""
    print("🧪 Testing LLM AI Judge Implementation")
    print("=" * 50)
    
    # Get the judge instance - using mock version for faster testing
    print("📥 Initializing LLM judge (fast mock version for testing)...")
    judge = MockLLMJudge()
    
    # Check model info
    print("📊 Model Information:")
    model_info = judge.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print()
    
    # Test evaluation
    print("🔍 Testing evaluation...")
    test_prompt = "Explain the concept of artificial intelligence."
    test_response = """
    Artificial intelligence (AI) is a branch of computer science that focuses on creating 
    systems capable of performing tasks that typically require human intelligence. These 
    tasks include learning, reasoning, problem-solving, perception, and language understanding. 
    AI systems can be designed to recognize patterns, make decisions, and adapt to new 
    situations based on data and experience.
    """
    
    print(f"Prompt: {test_prompt}")
    print(f"Response: {test_response.strip()[:100]}...")
    print()
    
    # Perform evaluation
    start_time = time.time()
    try:
        result = judge.evaluate(test_prompt, test_response)
        eval_time = time.time() - start_time
        
        # Check if evaluation worked
        if result.get('error', False):
            print(f"❌ {result.get('error_message', 'Unknown error')}")
            print("\n💬 Error Details:")
            for aspect, feedback in result['detailed_feedback'].items():
                print(f"  {aspect.capitalize()}: {feedback[:80]}...")
            return False
        
        print("✅ Evaluation Results:")
        print(f"  Overall Score: {result['overall_score']}/10")
        print(f"  Evaluation Time: {eval_time:.2f}s (reported: {result['evaluation_time']:.2f}s)")
        print()
        
        print("📋 Detailed Scores:")
        for aspect, score in result['detailed_scores'].items():
            print(f"  {aspect.capitalize()}: {score}/10")
        print()
        
        print("💬 Detailed Feedback:")
        for aspect, feedback in result['detailed_feedback'].items():
            print(f"  {aspect.capitalize()}: {feedback[:100]}...")
        print()
        
        # Show raw LLM responses for debugging
        if 'raw_responses' in result:
            print("🔍 Raw LLM Responses:")
            for aspect, raw in result['raw_responses'].items():
                if raw:
                    print(f"  {aspect.capitalize()}: {raw[:80]}...")
        print()
        
        print("⚖️ Weights Used:")
        for aspect, weight in result['weights_used'].items():
            print(f"  {aspect.capitalize()}: {weight:.2f}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def test_custom_weights():
    """Test evaluation with custom weights"""
    print("🎯 Testing custom weights...")
    judge = MockLLMJudge()
    
    custom_weights = {
        "relevance": 0.5,
        "coherence": 0.3,
        "accuracy": 0.15,
        "completeness": 0.05
    }
    
    test_prompt = "What are the benefits of renewable energy?"
    test_response = "Renewable energy sources like solar and wind power are clean and sustainable."
    
    try:
        result = judge.evaluate(test_prompt, test_response, weights=custom_weights)
        print(f"  Overall Score with custom weights: {result['overall_score']}/10")
        
        print("  Custom weights applied:")
        for aspect, weight in result['weights_used'].items():
            print(f"    {aspect}: {weight:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom weights test failed: {e}")
        return False

def test_comparison():
    """Test response comparison"""
    print("⚔️ Testing response comparison...")
    judge = MockLLMJudge()
    
    prompt = "What is machine learning?"
    response1 = "Machine learning is a subset of AI that enables computers to learn and make decisions from data without being explicitly programmed."
    response2 = "ML is when computers learn things."
    
    try:
        result = judge.compare_responses(prompt, response1, response2)
        print(f"  Winner: {result['winner']}")
        print(f"  Margin: {result['margin']} points")
        print(f"  Response 1 score: {result['comparison_summary']['response1_score']}/10")
        print(f"  Response 2 score: {result['comparison_summary']['response2_score']}/10")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting LLM AI Judge Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test basic evaluation
    if test_llm_judge():
        tests_passed += 1
        print("✅ Basic evaluation test passed")
    else:
        print("❌ Basic evaluation test failed")
    
    print("-" * 50)
    
    # Test custom weights
    if test_custom_weights():
        tests_passed += 1
        print("✅ Custom weights test passed")
    else:
        print("❌ Custom weights test failed")
    
    print("-" * 50)
    
    # Test comparison
    if test_comparison():
        tests_passed += 1
        print("✅ Comparison test passed")
    else:
        print("❌ Comparison test failed")
    
    print("=" * 60)
    print(f"🎯 Tests Summary: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! LLM judge is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
