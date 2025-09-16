#!/usr/bin/env python3
"""
Jina Judge Test Script
Test the new Jina-based judge implementation with real examples.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from models.jina_judge import get_jina_judge
import time

def test_jina_examples():
    """Test Jina judge with various examples"""
    print("🚀 Jina Judge Test - kaleinaNyan/jina-v3-rullmarena-judge")
    print("=" * 60)
    
    # Initialize the Jina judge
    print("📥 Initializing Jina judge...")
    judge = get_jina_judge()
    
    # Check model info
    print("📊 Model Information:")
    model_info = judge.get_model_info()
    for key, value in model_info.items():
        if isinstance(value, list):
            print(f"  {key}: {', '.join(value)}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Test cases with different quality levels
    test_cases = [
        {
            "name": "High Quality Response",
            "prompt": "Explain machine learning and its main types.",
            "response": """Machine learning is a subset of artificial intelligence that enables computers to learn and improve their performance on tasks through experience, without being explicitly programmed for each specific scenario.

There are three main types of machine learning:

1. **Supervised Learning**: Uses labeled training data to learn mappings between inputs and outputs. Examples include classification (predicting categories) and regression (predicting continuous values). Common algorithms include linear regression, decision trees, and neural networks.

2. **Unsupervised Learning**: Finds hidden patterns in data without labeled examples. This includes clustering (grouping similar data points), dimensionality reduction, and association rule learning. Examples include K-means clustering and principal component analysis.

3. **Reinforcement Learning**: Learns through trial and error by receiving rewards or penalties for actions taken in an environment. The agent learns to maximize cumulative rewards over time. This is used in game playing, robotics, and autonomous systems.

Each type serves different purposes and is suited for different kinds of problems, making machine learning a versatile tool for solving complex data-driven challenges."""
        },
        {
            "name": "Medium Quality Response",
            "prompt": "What are the benefits of renewable energy?",
            "response": """Renewable energy sources like solar, wind, and hydroelectric power offer several important benefits. They help reduce greenhouse gas emissions and combat climate change since they don't produce carbon dioxide during operation. Renewable energy also creates jobs in manufacturing, installation, and maintenance sectors. Additionally, they provide energy security by reducing dependence on imported fossil fuels and offer long-term cost stability since the fuel (sunlight, wind, water) is free."""
        },
        {
            "name": "Low Quality Response",
            "prompt": "How do vaccines work?",
            "response": "Vaccines help your body fight diseases. They make you immune to sickness. Doctors give them to kids and adults."
        },
        {
            "name": "Irrelevant Response",
            "prompt": "Explain photosynthesis in plants.",
            "response": "I really enjoy playing video games, especially strategy games like chess and civilization. My favorite hobby is cooking Italian food."
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
            
            # Show comparison details if available
            if 'comparison_details' in result:
                print("🔍 Comparison Analysis:")
                for quality_level, details in result['comparison_details'].items():
                    comparison = details['comparison']
                    score = details['derived_score']
                    print(f"   vs {quality_level.capitalize()}: {comparison['judgement']} → Score: {score}/10")
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

def test_jina_comparison():
    """Test Jina judge comparison functionality"""
    print(f"\n{'='*60}")
    print("🥊 JINA COMPARISON TEST")
    print(f"{'='*60}")
    
    judge = get_jina_judge()
    
    prompt = "What are the main causes of climate change?"
    
    response1 = """Climate change is primarily caused by human activities that increase greenhouse gas concentrations in the atmosphere. The main causes include:

1. **Fossil Fuel Combustion**: Burning coal, oil, and natural gas for electricity, heat, and transportation releases large amounts of CO2, the most significant greenhouse gas.

2. **Deforestation**: Cutting down forests reduces the Earth's capacity to absorb CO2, while burning or decomposing trees releases stored carbon.

3. **Industrial Processes**: Manufacturing, cement production, and chemical processes emit various greenhouse gases including CO2, methane, and nitrous oxide.

4. **Agriculture**: Livestock farming produces methane through digestion, while rice paddies and fertilizer use release methane and nitrous oxide.

5. **Transportation**: Cars, trucks, ships, and planes burn fossil fuels, contributing significantly to global emissions.

These activities have increased atmospheric CO2 levels by over 40% since pre-industrial times, leading to global warming and associated climate changes."""
    
    response2 = "Climate change happens because people burn fossil fuels and cut down trees. This makes the Earth warmer."
    
    print(f"📝 Prompt: {prompt}")
    print(f"💬 Response 1: {response1[:100]}...")
    print(f"💬 Response 2: {response2}")
    print()
    
    try:
        start_time = time.time()
        comparison = judge.compare_responses(prompt, response1, response2)
        comparison_time = time.time() - start_time
        
        if comparison.get('error'):
            print(f"❌ Comparison failed: {comparison['error_message']}")
            return
        
        print("🏆 Comparison Results:")
        print(f"   Winner: {comparison['winner']}")
        print(f"   Margin: {comparison['margin']} points")
        print(f"   Response 1 Score: {comparison['comparison_summary']['response1_score']}/10")
        print(f"   Response 2 Score: {comparison['comparison_summary']['response2_score']}/10")
        print(f"   Confidence: {comparison['judgement_details']['confidence']:.3f}")
        print(f"   Comparison Time: {comparison_time:.2f}s")
        print(f"   Judgement: {comparison['judgement_details']['judgement']}")
        print()
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

def test_jina_direct_comparison():
    """Test direct Jina model usage"""
    print(f"\n{'='*60}")
    print("🎯 DIRECT JINA MODEL TEST")
    print(f"{'='*60}")
    
    judge = get_jina_judge()
    
    # Test the direct comparison method
    prompt = "Explain the importance of sleep."
    response_a = "Sleep is crucial for physical and mental health, memory consolidation, immune function, and overall well-being. During sleep, the body repairs tissues and the brain processes information from the day."
    response_b = "Sleep is good for you and helps you feel better."
    
    print(f"📝 Prompt: {prompt}")
    print(f"🅰️  Response A: {response_a}")
    print(f"🅱️  Response B: {response_b}")
    print()
    
    try:
        start_time = time.time()
        result = judge._compare_responses_direct(prompt, response_a, response_b)
        comparison_time = time.time() - start_time
        
        if result.get('error'):
            print(f"❌ Direct comparison failed: {result['error']}")
            return
        
        print("🔥 Direct Jina Model Results:")
        print(f"   Judgement: {result['judgement']}")
        print(f"   Winner: {result['winner']}")
        print(f"   Score A: {result['score_a']}/10")
        print(f"   Score B: {result['score_b']}/10")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Comparison Time: {comparison_time:.2f}s")
        print(f"   Raw Logits: {[round(x, 3) for x in result['raw_logits']]}")
        
    except Exception as e:
        print(f"❌ Direct comparison failed: {e}")

def main():
    """Run Jina judge tests"""
    print("🎯 Starting Jina Judge Evaluation Tests")
    print("⚡ This model should be much faster and more reliable!")
    
    start_total = time.time()
    
    # Test individual examples
    test_jina_examples()
    
    # Test comparison
    test_jina_comparison()
    
    # Test direct model usage
    test_jina_direct_comparison()
    
    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"✅ All Jina tests completed in {total_time:.2f} seconds")
    print("🎉 Jina Judge evaluation finished!")

if __name__ == "__main__":
    main()
