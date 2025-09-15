#!/usr/bin/env python3
"""
Test Contextual Judges - VERSIÓN FINAL
=======================================

Prueba el sistema ContextualJudgesPanel que funciona sin transformers
y debe resolver el problema de scores fijos de 5.0/10.
"""

import asyncio
import sys
sys.path.append('app')

async def test_contextual_judges():
    """Prueba el sistema ContextualJudgesPanel con diferentes casos."""
    
    try:
        from models.contextual_judges import ContextualJudgesPanel
        
        print("🧠 Testing Contextual Judges Panel - VERSIÓN FINAL")
        print("=" * 60)
        
        # Crear panel
        panel = ContextualJudgesPanel(model_name="distilgpt2")
        
        # Test cases del problema original del usuario
        test_cases = [
            {
                "name": "🔴 CASO PROBLEMÁTICO ORIGINAL",
                "prompt": "chat, he agregado estas nuevas columnas: Domain Metodologia Fase Tags. Para Domain, Metodologia, y Fase, quiero que me des un prompt para que en mi web se pueda hacer un filtro por esas nuevas columnas. Ademas en la parte inferior de la card, aparezca la Domain (Tipo) y la o las fases. Y que al hacer la busqueda en el buscador, tambien la haga por estas tags. Se consiczo y punutal, lenguaje natural",
                "response": "¡Perfecto! Entonces te dejo un **prompt listo** para que se lo des a otra IA y te redacte la participación: Actúa como un ingeniero de sistemas apasionado por el Design Thinking y metodologías ágiles, pero que también es psicólogo con interés en los procesos humanos de los equipos. Redacta una participación para un foro académico respondiendo la consigna: ¿Cuál es la diferencia entre tener dos tracks (dos backlogs), vs. dos equipos (Dual Track Development is not Duel Track)?",
                "expected_overall": "2-4/10",
                "expected_relevance": "1-3/10",
                "should_detect": "BAJA RELEVANCIA - respuesta sobre Design Thinking cuando se pregunta sobre filtros web"
            },
            {
                "name": "🟢 CASO RELEVANTE",
                "prompt": "¿Qué es la inteligencia artificial y cómo funciona?",
                "response": "La inteligencia artificial es una rama de la ciencia de la computación que se enfoca en crear sistemas capaces de realizar tareas que tradicionalmente requieren inteligencia humana, como el aprendizaje, el reconocimiento de patrones, la toma de decisiones y el procesamiento del lenguaje natural. Funciona mediante algoritmos complejos que analizan datos para identificar patrones y hacer predicciones.",
                "expected_overall": "7-9/10", 
                "expected_relevance": "8-9/10",
                "should_detect": "ALTA RELEVANCIA - respuesta directa y completa sobre IA"
            },
            {
                "name": "🟡 CASO PARCIALMENTE RELEVANTE",
                "prompt": "Explica cómo programar en Python para principiantes",
                "response": "Python es un lenguaje de programación muy popular y fácil de aprender. Los lenguajes de programación son herramientas importantes para el desarrollo de software en la actualidad.",
                "expected_overall": "4-6/10",
                "expected_relevance": "4-6/10", 
                "should_detect": "RELEVANCIA PARCIAL - menciona Python pero no explica cómo programar"
            }
        ]
        
        print("\n🧪 EJECUTANDO TESTS:")
        print("-" * 40)
        
        results = []
        all_passed = True
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}")
            print("   " + "="*50)
            
            # Realizar evaluación
            start_time = asyncio.get_event_loop().time()
            result = await panel.evaluate_response_async(
                test_case["prompt"],
                test_case["response"]
            )
            end_time = asyncio.get_event_loop().time()
            eval_time = end_time - start_time
            
            results.append({
                'test_case': test_case,
                'result': result,
                'eval_time': eval_time
            })
            
            # Mostrar resultados detallados
            overall_score = result['overall_score']
            relevance_score = result['individual_scores']['relevance']['score']
            consensus = result['consensus']
            semantic_overlap = result['evaluation_stats']['semantic_overlap']
            
            print(f"   📊 Score general: {overall_score}/10")
            print(f"   🎯 Relevancia: {relevance_score}/10")
            print(f"   🤝 Consenso: {consensus}%")
            print(f"   📈 Overlap semántico: {semantic_overlap:.1%}")
            print(f"   ⏱️  Tiempo: {eval_time:.3f}s")
            print(f"   🤖 Modelo: {result['model_used']}")
            
            # Mostrar feedback de relevancia
            relevance_feedback = result['individual_scores']['relevance']['feedback']
            print(f"   💬 Feedback: {relevance_feedback}")
            
            # Mostrar aspectos individuales
            print("   📋 Scores detallados:")
            for aspect, aspect_result in result['individual_scores'].items():
                score = aspect_result['score']
                print(f"      {aspect.capitalize()}: {score}/10")
            
            # Verificar si pasó el test
            expected_range = test_case['expected_overall'].split('-')
            min_expected = float(expected_range[0])
            max_expected = float(expected_range[1].replace('/10', ''))
            
            test_passed = min_expected <= overall_score <= max_expected
            
            if test_passed:
                print(f"   ✅ TEST PASADO - Score en rango esperado ({test_case['expected_overall']})")
            else:
                print(f"   ❌ TEST FALLIDO - Score {overall_score}/10 fuera del rango {test_case['expected_overall']}")
                all_passed = False
            
            print(f"   🔍 Detección: {test_case['should_detect']}")
        
        # Resumen comparativo final
        print("\n" + "=" * 70)
        print("📊 RESUMEN COMPARATIVO - ANTES VS DESPUÉS")
        print("=" * 70)
        
        print(f"\n{'Test Case':<30} | {'Antes':<10} | {'Ahora':<10} | {'Status':<15}")
        print("-" * 70)
        
        for i, result_data in enumerate(results):
            test_name = result_data['test_case']['name'][:29]
            before_score = "5.0/10"  # El problema original - siempre 5.0
            current_score = f"{result_data['result']['overall_score']}/10"
            
            # Determinar status
            current_relevance = result_data['result']['individual_scores']['relevance']['score']
            if "PROBLEMÁTICO" in test_name.upper() and current_relevance <= 4:
                status = "🎯 CORREGIDO"
            elif "RELEVANTE" in test_name.upper() and current_relevance >= 7:
                status = "✅ CORRECTO"
            elif "PARCIAL" in test_name.upper() and 4 <= current_relevance <= 6:
                status = "✅ CORRECTO"
            else:
                status = "⚠️  REVISAR"
            
            print(f"{test_name:<30} | {before_score:<10} | {current_score:<10} | {status:<15}")
        
        # Análisis final
        print(f"\n🎯 ANÁLISIS FINAL:")
        print("-" * 30)
        if all_passed:
            print("✅ TODOS LOS TESTS PASARON")
            print("🎉 El problema de scores fijos de 5.0/10 está SOLUCIONADO")
            print("🧠 El sistema ahora evalúa correctamente el contexto")
        else:
            print("⚠️  ALGUNOS TESTS FALLARON - revisar implementación")
        
        # Verificación específica del problema original
        original_problem = results[0]  # Primer caso - el problemático
        original_relevance = original_problem['result']['individual_scores']['relevance']['score']
        
        print(f"\n🔍 VERIFICACIÓN DEL PROBLEMA ORIGINAL:")
        print(f"   Relevancia antes: 5.0/10 (incorrecto)")
        print(f"   Relevancia ahora: {original_relevance}/10")
        
        if original_relevance <= 4:
            print("   🎯 PROBLEMA SOLUCIONADO - Detecta correctamente baja relevancia")
        else:
            print("   ❌ PROBLEMA PERSISTE - Sigue dando score alto")
        
        return results, all_passed
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return None, False

async def main():
    """Función principal."""
    print("🏛️ AI Judges Panel - Test Contextual System")
    print("=" * 70)
    print("🎯 Objetivo: Verificar que se solucionó el problema de scores fijos 5.0/10")
    print("=" * 70)
    
    # Test principal
    results, all_passed = await test_contextual_judges()
    
    print("\n" + "=" * 70)
    if results and all_passed:
        print("🎉 ¡ÉXITO TOTAL! El sistema ContextualJudges funciona correctamente")
        print("✅ Ya no más scores fijos de 5.0/10")
        print("🧠 Evaluación contextual inteligente implementada")
        print("⚡ Sistema ultra-rápido sin dependencias pesadas")
    else:
        print("❌ Hay problemas con el sistema - revisar implementación")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
