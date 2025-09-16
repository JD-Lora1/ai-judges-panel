#!/usr/bin/env python3
"""
Simple Judge Test Script
Test the reliable sentence transformer based judge implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from models.simple_judge import get_simple_judge
import time

def test_simple_examples():
    """Test simple judge with various examples"""
    print("🚀 Simple Judge Test - Sentence Transformer Based")
    print("=" * 60)
    
    # Initialize the simple judge
    print("📥 Initializing Simple judge...")
    judge = get_simple_judge()
    
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
        },
        {
            "name": "Empty/Very Short Response",
            "prompt": "Describe the process of cellular respiration.",
            "response": "Yes."
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

def test_simple_comparison():
    """Test simple judge comparison functionality"""
    print(f"\n{'='*60}")
    print("🥊 SIMPLE JUDGE COMPARISON TEST")
    print(f"{'='*60}")
    
    judge = get_simple_judge()
    
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
            print(f"❌ Comparison failed")
            return
        
        print("🏆 Comparison Results:")
        print(f"   Winner: {comparison['winner']}")
        print(f"   Margin: {comparison['margin']} points")
        print(f"   Response 1 Score: {comparison['comparison_summary']['response1_score']}/10")
        print(f"   Response 2 Score: {comparison['comparison_summary']['response2_score']}/10")
        print(f"   Comparison Time: {comparison_time:.2f}s")
        print()
        
        # Show detailed breakdown
        print("📊 Detailed Comparison:")
        eval1 = comparison['response1_evaluation']
        eval2 = comparison['response2_evaluation']
        
        aspects = ['relevance', 'coherence', 'accuracy', 'completeness']
        for aspect in aspects:
            score1 = eval1['detailed_scores'][aspect]
            score2 = eval2['detailed_scores'][aspect] 
            winner = "Response 1" if score1 > score2 else "Response 2" if score2 > score1 else "Tie"
            print(f"   {aspect.capitalize()}: {score1} vs {score2} → {winner}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

def test_custom_weights():
    """Test custom weights functionality"""
    print(f"\n{'='*60}")
    print("🎯 CUSTOM WEIGHTS TEST")
    print(f"{'='*60}")
    
    judge = get_simple_judge()
    
    # Test with different weight priorities
    weight_scenarios = [
        {
            "name": "Relevance Priority",
            "weights": {"relevance": 0.6, "coherence": 0.2, "accuracy": 0.1, "completeness": 0.1}
        },
        {
            "name": "Completeness Priority", 
            "weights": {"relevance": 0.2, "coherence": 0.2, "accuracy": 0.2, "completeness": 0.4}
        },
        {
            "name": "Balanced (Default)",
            "weights": None  # Will use default weights
        }
    ]
    
    prompt = "Explain the water cycle."
    response = """The water cycle is the continuous movement of water on, above, and below Earth's surface. It includes evaporation from oceans and lakes, condensation into clouds, precipitation as rain or snow, and collection in bodies of water. This process is powered by solar energy and gravity, and it's essential for distributing fresh water across the planet."""
    
    print(f"📝 Prompt: {prompt}")
    print(f"💬 Response: {response}")
    print()
    
    for scenario in weight_scenarios:
        print(f"🎯 Testing: {scenario['name']}")
        
        try:
            result = judge.evaluate(prompt, response, weights=scenario['weights'])
            
            if result.get('error'):
                print(f"   ❌ Error: {result.get('error_message')}")
                continue
            
            print(f"   Overall Score: {result['overall_score']}/10")
            print(f"   Weights Used: {result['weights_used']}")
            print()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def main():
    """Run simple judge tests"""
    print("🎯 Starting Simple Judge Evaluation Tests")
    print("⚡ This approach should be fast, reliable, and dependency-free!")
    
    start_total = time.time()
    
    # Test individual examples
    test_simple_examples()
    
    # Test comparison
    test_simple_comparison()
    
    # Test custom weights
    test_custom_weights()
    
    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"✅ All Simple Judge tests completed in {total_time:.2f} seconds")
    print("🎉 Simple Judge evaluation finished!")

if __name__ == "__main__":
    main()
