#!/usr/bin/env python3
"""
Test script for Phi-2 AI Judge
Verifies that the Phi-2 model implementation works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from models.phi2_judge import get_phi2_judge
import time

def test_phi2_judge():
    """Test the Phi-2 judge functionality"""
    print("🧪 Testing Phi-2 AI Judge Implementation")
    print("=" * 50)
    
    # Get the judge instance
    print("📥 Initializing Phi-2 judge...")
    judge = get_phi2_judge()
    
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
            print(f"  {aspect.capitalize()}: {feedback[:80]}...")
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
    judge = get_phi2_judge()
    
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
    judge = get_phi2_judge()
    
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
    print("🚀 Starting Phi-2 AI Judge Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test basic evaluation
    if test_phi2_judge():
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
        print("🎉 All tests passed! Phi-2 judge is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
