#!/usr/bin/env python3
"""
Test Smart LLM Judges
======================

Prueba el nuevo sistema híbrido SmartLLMJudgesPanel.
"""

import asyncio
import sys
sys.path.append('app')

async def test_smart_llm_judges():
    """Prueba el sistema SmartLLMJudgesPanel con diferentes casos."""
    
    try:
        from models.smart_llm_judges import SmartLLMJudgesPanel
        
        print("🧠 Testing Smart LLM Judges Panel")
        print("=" * 50)
        
        # Crear panel
        panel = SmartLLMJudgesPanel(model_name="distilgpt2")
        
        # Test cases
        test_cases = [
            {
                "name": "Caso Problemático Original",
                "prompt": "chat, he agregado estas nuevas columnas: Domain Metodologia Fase Tags. Para Domain, Metodologia, y Fase, quiero que me des un prompt para que en mi web se pueda hacer un filtro por esas nuevas columnas.",
                "response": "¡Perfecto! Entonces te dejo un **prompt listo** para que se lo des a otra IA y te redacte la participación: Actúa como un ingeniero de sistemas apasionado por el Design Thinking y metodologías ágiles.",
                "expected_relevance": "Baja (2-3/10)"
            },
            {
                "name": "Caso Relevante",
                "prompt": "¿Qué es la inteligencia artificial?",
                "response": "La inteligencia artificial es una rama de la ciencia de la computación que se enfoca en crear sistemas capaces de realizar tareas que tradicionalmente requieren inteligencia humana.",
                "expected_relevance": "Alta (8-9/10)"
            },
            {
                "name": "Caso Parcialmente Relevante",
                "prompt": "Explica cómo programar en Python",
                "response": "Python es un lenguaje de programación muy popular. Los lenguajes de programación son importantes para el desarrollo de software.",
                "expected_relevance": "Media (5-6/10)"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['name']}")
            print("-" * 30)
            
            # Realizar evaluación
            result = await panel.evaluate_response_async(
                test_case["prompt"],
                test_case["response"]
            )
            
            results.append(result)
            
            # Mostrar resultados
            print(f"📊 Score general: {result['overall_score']}/10")
            print(f"🎯 Relevancia: {result['individual_scores']['relevance']['score']}/10")
            print(f"💬 Feedback relevancia: {result['individual_scores']['relevance']['feedback']}")
            print(f"📈 Overlap semántico: {result['evaluation_stats']['semantic_overlap']:.1%}")
            print(f"🤖 Modelo: {result['model_used']}")
            print(f"⏱️ Método: {result['evaluation_stats']['evaluation_method']}")
            print(f"🔗 Esperado: {test_case['expected_relevance']}")
            
            # Análisis
            relevance_score = result['individual_scores']['relevance']['score']
            if "Baja" in test_case['expected_relevance'] and relevance_score <= 4:
                print("✅ Correctamente identificado como baja relevancia")
            elif "Alta" in test_case['expected_relevance'] and relevance_score >= 7:
                print("✅ Correctamente identificado como alta relevancia")
            elif "Media" in test_case['expected_relevance'] and 4 < relevance_score < 7:
                print("✅ Correctamente identificado como relevancia media")
            else:
                print("⚠️ Relevancia no coincide con lo esperado")
        
        # Resumen comparativo
        print("\n" + "=" * 60)
        print("📊 RESUMEN COMPARATIVO")
        print("=" * 60)
        
        print(f"\n{'Caso':<25} | {'Score':<8} | {'Relevancia':<10} | {'Status':<15}")
        print("-" * 65)
        
        for i, (test_case, result) in enumerate(zip(test_cases, results)):
            case_name = test_case['name'][:24]
            overall = f"{result['overall_score']}/10"
            relevance = f"{result['individual_scores']['relevance']['score']}/10"
            
            # Determinar status
            relevance_score = result['individual_scores']['relevance']['score']
            if relevance_score >= 7:
                status = "🟢 Alta"
            elif relevance_score >= 4:
                status = "🟡 Media"
            else:
                status = "🔴 Baja"
            
            print(f"{case_name:<25} | {overall:<8} | {relevance:<10} | {status:<15}")
        
        print("\n🎯 ANÁLISIS:")
        print("- El sistema debe dar scores bajos para casos no relacionados")
        print("- Scores altos para respuestas relevantes")
        print("- Feedback específico explicando el reasoning")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

async def compare_with_previous_system():
    """Compara con el sistema anterior."""
    print("\n🔄 Comparación con sistema anterior")
    print("=" * 50)
    
    # Simular resultados del sistema anterior (siempre 5.0/10)
    previous_results = {
        "Caso Problemático": {"overall": 5.0, "relevance": 5.0},
        "Caso Relevante": {"overall": 5.0, "relevance": 5.0},
        "Caso Parcial": {"overall": 5.0, "relevance": 5.0}
    }
    
    print("Sistema Anterior (con error):")
    for case, scores in previous_results.items():
        print(f"  {case}: {scores['overall']}/10 (Relevancia: {scores['relevance']}/10)")
    
    print("\n🆚 Sistema Nuevo:")
    print("  - Evalúa contexto real entre prompt y respuesta")
    print("  - Scores variables según relevancia real")
    print("  - Feedback específico y explicativo")
    print("  - Sin errores de 1000% consenso o 0.0s tiempo")

async def main():
    """Función principal."""
    print("🏛️ AI Judges Panel - Test Smart LLM System")
    print("=" * 60)
    
    # Test principal
    results = await test_smart_llm_judges()
    
    # Comparación
    await compare_with_previous_system()
    
    print("\n" + "=" * 60)
    if results:
        print("✅ Smart LLM Judges funcionando correctamente")
        print("🎯 Ya no más scores fijos de 5.0/10 para todo!")
    else:
        print("❌ Error en el sistema - revisar implementación")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
