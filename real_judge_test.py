#!/usr/bin/env python3
"""
Real LLM Judge Test - Using actual vwxyzjn/online_dpo_llmjudge model
Shows real scores and evaluations for example prompts and responses.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from models.phi2_judge import get_phi2_judge
import time

def test_real_examples():
    """Test with real examples using the actual LLM judge model"""
    print("🔥 Real LLM Judge Test - vwxyzjn/online_dpo_llmjudge")
    print("=" * 60)
    
    # Initialize the real judge
    print("📥 Loading actual LLM judge model...")
    print("⚠️  This may take a few minutes to download the model...")
    judge = get_phi2_judge()
    
    # Test cases with different quality levels
    test_cases = [
        {
            "name": "High Quality Response",
            "prompt": "Explain the concept of machine learning.",
            "response": """Machine learning is a subset of artificial intelligence that enables computers to learn and improve their performance on tasks through experience, without being explicitly programmed for each specific scenario. 

It works by training algorithms on large datasets, allowing them to identify patterns and make predictions or decisions on new, unseen data. There are three main types:

1. Supervised Learning: Uses labeled training data to learn mappings between inputs and outputs
2. Unsupervised Learning: Finds hidden patterns in data without labeled examples  
3. Reinforcement Learning: Learns through trial and error by receiving rewards or penalties

Common applications include recommendation systems, image recognition, natural language processing, and autonomous vehicles. The key advantage is that ML systems can adapt and improve over time as they process more data."""
        },
        {
            "name": "Medium Quality Response", 
            "prompt": "What is climate change?",
            "response": """Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, scientific evidence shows that human activities since the 1800s have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas. This releases greenhouse gases into the atmosphere that trap heat from the sun."""
        },
        {
            "name": "Low Quality Response",
            "prompt": "How do computers work?",
            "response": "Computers use electricity and have chips inside. They can do math fast and store information. You can use them for games and internet."
        },
        {
            "name": "Irrelevant Response",
            "prompt": "Explain photosynthesis in plants.",
            "response": "I like pizza a lot. My favorite toppings are pepperoni and mushrooms. Pizza is really good for dinner and sometimes breakfast too."
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*60}")
        print(f"📝 Prompt: {test_case['prompt']}")
        print(f"💬 Response: {test_case['response'][:100]}...")
        print()
        
        start_time = time.time()
        try:
            result = judge.evaluate(test_case['prompt'], test_case['response'])
            eval_time = time.time() - start_time
            
            if result.get('error', False):
                print(f"❌ Error: {result.get('error_message', 'Unknown error')}")
                continue
            
            # Display results
            print(f"⭐ Overall Score: {result['overall_score']}/10")
            print(f"⏱️  Evaluation Time: {eval_time:.2f}s")
            print()
            
            print("📊 Detailed Scores:")
            for aspect, score in result['detailed_scores'].items():
                print(f"   {aspect.capitalize()}: {score}/10")
            print()
            
            print("💭 Detailed Feedback:")
            for aspect, feedback in result['detailed_feedback'].items():
                print(f"   {aspect.capitalize()}: {feedback}")
            print()
            
            # Store result for summary
            results.append({
                'name': test_case['name'],
                'overall_score': result['overall_score'],
                'detailed_scores': result['detailed_scores'],
                'eval_time': eval_time
            })
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            continue
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("📋 SUMMARY OF RESULTS")
        print(f"{'='*60}")
        
        for result in results:
            print(f"{result['name']:25} | Overall: {result['overall_score']:5.2f}/10 | Time: {result['eval_time']:6.2f}s")
        
        print(f"\n📈 Score Distribution:")
        scores = [r['overall_score'] for r in results]
        print(f"   Highest Score: {max(scores):.2f}/10")
        print(f"   Lowest Score:  {min(scores):.2f}/10")
        print(f"   Average Score: {sum(scores)/len(scores):.2f}/10")

def test_comparison_example():
    """Test response comparison with real model"""
    print(f"\n{'='*60}")
    print("🥊 RESPONSE COMPARISON TEST")
    print(f"{'='*60}")
    
    judge = get_phi2_judge()
    
    prompt = "What are the benefits of renewable energy?"
    
    response1 = """Renewable energy offers numerous benefits including environmental, economic, and social advantages. 
    
    Environmental benefits:
    - Significantly reduces greenhouse gas emissions and carbon footprint
    - Minimizes air and water pollution compared to fossil fuels
    - Helps combat climate change and global warming
    
    Economic benefits:
    - Creates jobs in manufacturing, installation, and maintenance
    - Reduces dependence on imported fossil fuels
    - Provides long-term cost stability as fuel costs are eliminated
    - Stimulates local economic development
    
    Social benefits:
    - Improves public health by reducing pollution-related diseases
    - Increases energy security and independence
    - Provides electricity access to remote areas
    
    The transition to renewable energy is essential for sustainable development and ensuring a cleaner future for generations to come."""
    
    response2 = "Renewable energy is good because it's clean and doesn't pollute like coal. It also creates some jobs and can save money."
    
    print(f"📝 Prompt: {prompt}")
    print(f"💬 Response 1: {response1[:100]}...")
    print(f"💬 Response 2: {response2}")
    print()
    
    try:
        comparison = judge.compare_responses(prompt, response1, response2)
        
        print("🏆 Comparison Results:")
        print(f"   Winner: {comparison['winner']}")
        print(f"   Margin: {comparison['margin']} points")
        print(f"   Response 1 Score: {comparison['comparison_summary']['response1_score']}/10")
        print(f"   Response 2 Score: {comparison['comparison_summary']['response2_score']}/10")
        print()
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

def main():
    """Run real LLM judge tests"""
    print("🎯 Starting Real LLM Judge Evaluation")
    print("⚠️  Note: This will download and use the actual model (may take time)")
    
    # Ask user if they want to continue
    response = input("\n🤔 Continue with real model test? This may take several minutes... (y/n): ")
    if response.lower() != 'y':
        print("❌ Test cancelled.")
        return
    
    start_total = time.time()
    
    # Test individual examples
    test_real_examples()
    
    # Test comparison
    test_comparison_example()
    
    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"✅ All tests completed in {total_time:.2f} seconds")
    print("🎉 Real LLM Judge evaluation finished!")

if __name__ == "__main__":
    main()
