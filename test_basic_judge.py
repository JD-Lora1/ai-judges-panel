#!/usr/bin/env python3
"""
Basic Judge Test Script
Test the dependency-free rule-based judge implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from models.basic_judge import get_basic_judge
import time

def test_basic_examples():
    """Test basic judge with various examples"""
    print("🚀 Basic Judge Test - Pure Python Text Analysis")
    print("=" * 60)
    
    # Initialize the basic judge
    print("📥 Initializing Basic judge...")
    judge = get_basic_judge()
    
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

1. Supervised Learning: Uses labeled training data to learn mappings between inputs and outputs. Examples include classification (predicting categories) and regression (predicting continuous values). Common algorithms include linear regression, decision trees, and neural networks.

2. Unsupervised Learning: Finds hidden patterns in data without labeled examples. This includes clustering (grouping similar data points), dimensionality reduction, and association rule learning. Examples include K-means clustering and principal component analysis.

3. Reinforcement Learning: Learns through trial and error by receiving rewards or penalties for actions taken in an environment. The agent learns to maximize cumulative rewards over time. This is used in game playing, robotics, and autonomous systems.

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
        },
        {
            "name": "Technical Response with Numbers",
            "prompt": "What is the speed of light?",
            "response": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 186,282 miles per second). This fundamental constant, denoted as 'c', is a cornerstone of Einstein's theory of relativity and represents the maximum speed at which information can travel through space."
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
            print(f"⏱️  Evaluation Time: {eval_time:.3f}s")
            print(f"📏 Word Count: {result['input_info']['word_count']} words")
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
                'eval_time': eval_time,
                'word_count': result['input_info']['word_count']
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
            print(f"{result['name']:30} | Score: {result['overall_score']:5.2f}/10 | Words: {result['word_count']:3d} | Time: {result['eval_time']:6.3f}s")
        
        print(f"\n📈 Score Distribution:")
        scores = [r['overall_score'] for r in results]
        print(f"   Highest Score: {max(scores):.2f}/10")
        print(f"   Lowest Score:  {min(scores):.2f}/10") 
        print(f"   Average Score: {sum(scores)/len(scores):.2f}/10")
        print(f"   Score Range: {max(scores) - min(scores):.2f} points")

def test_basic_comparison():
    """Test basic judge comparison functionality"""
    print(f"\n{'='*60}")
    print("🥊 BASIC JUDGE COMPARISON TEST")
    print(f"{'='*60}")
    
    judge = get_basic_judge()
    
    comparison_tests = [
        {
            "name": "Detailed vs Simple",
            "prompt": "What are the main causes of climate change?",
            "response1": """Climate change is primarily caused by human activities that increase greenhouse gas concentrations in the atmosphere. The main causes include:

1. Fossil Fuel Combustion: Burning coal, oil, and natural gas for electricity, heat, and transportation releases large amounts of CO2, the most significant greenhouse gas.

2. Deforestation: Cutting down forests reduces the Earth's capacity to absorb CO2, while burning or decomposing trees releases stored carbon.

3. Industrial Processes: Manufacturing, cement production, and chemical processes emit various greenhouse gases including CO2, methane, and nitrous oxide.

4. Agriculture: Livestock farming produces methane through digestion, while rice paddies and fertilizer use release methane and nitrous oxide.

5. Transportation: Cars, trucks, ships, and planes burn fossil fuels, contributing significantly to global emissions.

These activities have increased atmospheric CO2 levels by over 40% since pre-industrial times, leading to global warming and associated climate changes.""",
            "response2": "Climate change happens because people burn fossil fuels and cut down trees. This makes the Earth warmer."
        },
        {
            "name": "Relevant vs Irrelevant",
            "prompt": "How does photosynthesis work?",
            "response1": "Photosynthesis is the process by which plants convert sunlight into energy. Chlorophyll absorbs light energy, which is used to convert carbon dioxide and water into glucose and oxygen.",
            "response2": "I like pizza and video games. My favorite color is blue and I enjoy watching movies on weekends."
        }
    ]
    
    for test in comparison_tests:
        print(f"\n🔍 Test: {test['name']}")
        print(f"📝 Prompt: {test['prompt']}")
        print(f"💬 Response 1: {test['response1'][:80]}...")
        print(f"💬 Response 2: {test['response2'][:80]}...")
        print()
        
        try:
            start_time = time.time()
            comparison = judge.compare_responses(test['prompt'], test['response1'], test['response2'])
            comparison_time = time.time() - start_time
            
            if comparison.get('error'):
                print(f"   ❌ Comparison failed")
                continue
            
            print("🏆 Comparison Results:")
            print(f"   Winner: {comparison['winner']}")
            print(f"   Margin: {comparison['margin']:.2f} points")
            print(f"   Response 1 Score: {comparison['comparison_summary']['response1_score']}/10")
            print(f"   Response 2 Score: {comparison['comparison_summary']['response2_score']}/10")
            print(f"   Comparison Time: {comparison_time:.3f}s")
            
            # Detailed breakdown
            print("\n📊 Detailed Comparison:")
            eval1 = comparison['response1_evaluation']
            eval2 = comparison['response2_evaluation']
            
            aspects = ['relevance', 'coherence', 'accuracy', 'completeness']
            for aspect in aspects:
                score1 = eval1['detailed_scores'][aspect]
                score2 = eval2['detailed_scores'][aspect]
                winner = "Response 1" if score1 > score2 else "Response 2" if score2 > score1 else "Tie"
                diff = abs(score1 - score2)
                print(f"   {aspect.capitalize():12}: {score1:4.1f} vs {score2:4.1f} (Δ{diff:4.1f}) → {winner}")
            
        except Exception as e:
            print(f"   ❌ Comparison failed: {e}")

def test_edge_cases():
    """Test edge cases and special scenarios"""
    print(f"\n{'='*60}")
    print("🔬 EDGE CASES TEST")
    print(f"{'='*60}")
    
    judge = get_basic_judge()
    
    edge_cases = [
        {
            "name": "Empty Response",
            "prompt": "What is artificial intelligence?", 
            "response": ""
        },
        {
            "name": "Single Word Response",
            "prompt": "Is the sky blue?",
            "response": "Yes."
        },
        {
            "name": "Very Long Response (Repetitive)",
            "prompt": "What is water?",
            "response": "Water is water. Water is H2O. Water is water. Water is liquid. Water is water. " * 20
        },
        {
            "name": "Numbers and Technical Terms",
            "prompt": "What is the formula for water?",
            "response": "The chemical formula for water is H2O, which means each molecule contains 2 hydrogen atoms and 1 oxygen atom. The molecular weight is approximately 18.015 g/mol."
        }
    ]
    
    for test_case in edge_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print(f"📝 Prompt: {test_case['prompt']}")
        print(f"💬 Response: '{test_case['response'][:50]}{'...' if len(test_case['response']) > 50 else ''}'")
        
        try:
            result = judge.evaluate(test_case['prompt'], test_case['response'])
            
            print(f"   Overall Score: {result['overall_score']}/10")
            print(f"   Word Count: {result['input_info']['word_count']}")
            print(f"   Evaluation Time: {result['evaluation_time']:.3f}s")
            
            if result.get('error'):
                print(f"   Error: {result.get('error_message')}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def main():
    """Run basic judge tests"""
    print("🎯 Starting Basic Judge Evaluation Tests")
    print("⚡ Zero dependencies - Pure Python text analysis!")
    print("🔧 Rule-based approach with linguistic heuristics")
    
    start_total = time.time()
    
    # Test individual examples
    test_basic_examples()
    
    # Test comparison
    test_basic_comparison()
    
    # Test edge cases
    test_edge_cases()
    
    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"✅ All Basic Judge tests completed in {total_time:.2f} seconds")
    print("🎉 Basic Judge evaluation finished!")
    print("💡 This approach works entirely offline with no model downloads!")

if __name__ == "__main__":
    main()
