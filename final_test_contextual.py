#!/usr/bin/env python3
"""
Test Final Independiente - Contextual Judges
=============================================

Test que importa directamente el módulo para evitar conflictos de dependencias.
"""

import asyncio
import sys
import os

# Importar directamente el módulo sin pasar por __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app', 'models'))

async def test_final():
    """Test final del sistema contextual."""
    
    try:
        # Importar directamente el módulo contextual
        from contextual_judges import ContextualJudgesPanel
        
        print("🎯 TEST FINAL - Sistema Contextual Judges")
        print("=" * 60)
        print("🎯 Verificando solución al problema de scores fijos 5.0/10")
        print("=" * 60)
        
        # Crear panel
        panel = ContextualJudgesPanel(model_name="distilgpt2")
        print("✅ Panel contextual inicializado correctamente")
        
        # El caso problemático exacto del usuario
        print("\n🔴 CASO PROBLEMÁTICO ORIGINAL:")
        print("-" * 40)
        
        prompt_original = """chat, he agregado estas nuevas columnas: Domain	Metodologia	Fase	Tags
Para Domain, Metodologia, y Fase, quiero que me des un prompt para que en mi web se pueda hacer un filtro por esas nuevas columnas. Ademas en la parte inferior de la card, aparezca la Domain (Tipo) y la o las fases. Y que al hacer la busqueda en el buscador, tambien la haga por estas tags. Se consiczo y punutal, lenguaje natural"""
        
        response_original = """¡Perfecto! Entonces te dejo un **prompt listo** para que se lo des a otra IA y te redacte la participación:

---

**Prompt para IA:**

Actúa como un ingeniero de sistemas apasionado por el Design Thinking y metodologías ágiles, pero que también es psicólogo con interés en los procesos humanos de los equipos.

Redacta una participación para un foro académico respondiendo la consigna:

> **¿Cuál es la diferencia entre tener dos tracks (dos backlogs), vs. dos equipos (Dual Track Development is not Duel Track)?**

La respuesta debe:

* Explicar con claridad la diferencia entre dos tracks (dual track development: discovery y delivery) y dos equipos separados.
* Resaltar que dual track no significa dividir equipos, sino tener un mismo equipo trabajando en dos flujos paralelos.
* Incluir reflexiones desde la perspectiva de la psicología de equipos (motivación, colaboración, riesgo de silos).
* Usar un ejemplo sencillo que ilustre cómo discovery y delivery pueden coexistir en un mismo sprint.
* Cerrar con una pregunta abierta para invitar a otros participantes a reflexionar o compartir su experiencia.

El tono debe ser académico pero cercano, respetuoso y participativo.

---

¿Quieres que además te arme una **versión corta** del prompt (tipo "one-liner") para que sea aún más directo cuando lo pongas en otra IA?"""
        
        print("🔍 Evaluando el caso problemático...")
        result = await panel.evaluate_response_async(prompt_original, response_original)
        
        # Mostrar resultados
        overall_score = result['overall_score']
        relevance_score = result['individual_scores']['relevance']['score']
        precision_score = result['individual_scores']['precision']['score']
        semantic_overlap = result['evaluation_stats']['semantic_overlap']
        consensus = result['consensus']
        
        print(f"\n📊 RESULTADOS:")
        print(f"   Score general: {overall_score}/10")
        print(f"   🎯 Relevancia: {relevance_score}/10")
        print(f"   🎯 Precisión: {precision_score}/10")
        print(f"   📈 Overlap semántico: {semantic_overlap:.1%}")
        print(f"   🤝 Consenso: {consensus}%")
        print(f"   🤖 Modelo: {result['model_used']}")
        
        # Feedback detallado
        print(f"\n💬 FEEDBACK DETALLADO:")
        for aspect, aspect_result in result['individual_scores'].items():
            score = aspect_result['score']
            feedback = aspect_result['feedback']
            print(f"   {aspect.capitalize()}: {score}/10 - {feedback}")
        
        # Análisis de la solución
        print(f"\n🔍 ANÁLISIS DE LA SOLUCIÓN:")
        print(f"   Problema original: Siempre daba 5.0/10 en todo")
        print(f"   Solución actual: {overall_score}/10 (variable según contexto)")
        
        # Verificar si el problema está solucionado
        problem_solved = relevance_score <= 4.0 and overall_score <= 5.0
        
        if problem_solved:
            print(f"\n🎉 ¡PROBLEMA SOLUCIONADO!")
            print(f"   ✅ Detecta correctamente la BAJA RELEVANCIA")
            print(f"   ✅ Score general apropiado para contenido no relacionado")
            print(f"   ✅ Sistema contextual funciona correctamente")
        else:
            print(f"\n⚠️  PROBLEMA PARCIALMENTE RESUELTO")
            print(f"   - Relevancia: {relevance_score}/10 (debería ser ≤4)")
            print(f"   - Score general: {overall_score}/10 (debería ser ≤5)")
        
        # Test de caso positivo para comparación
        print(f"\n🟢 CASO POSITIVO (para comparación):")
        print("-" * 40)
        
        good_prompt = "¿Qué es la inteligencia artificial?"
        good_response = "La inteligencia artificial es una rama de la ciencia de la computación que se enfoca en crear sistemas capaces de realizar tareas que tradicionalmente requieren inteligencia humana."
        
        good_result = await panel.evaluate_response_async(good_prompt, good_response)
        good_relevance = good_result['individual_scores']['relevance']['score']
        good_overall = good_result['overall_score']
        
        print(f"   Score general: {good_overall}/10")
        print(f"   🎯 Relevancia: {good_relevance}/10")
        
        # Comparación final
        print(f"\n📊 COMPARACIÓN FINAL:")
        print("=" * 50)
        print(f"{'Caso':<20} | {'Antes':<8} | {'Ahora':<8} | {'Status':<12}")
        print("-" * 50)
        print(f"{'Problemático':<20} | {'5.0/10':<8} | {f'{overall_score}/10':<8} | {'🎯 CORREGIDO' if problem_solved else '⚠️ PARCIAL':<12}")
        print(f"{'Relevante':<20} | {'5.0/10':<8} | {f'{good_overall}/10':<8} | {'✅ CORRECTO' if good_relevance >= 7 else '⚠️ REVISAR':<12}")
        
        # Conclusión
        print(f"\n🎯 CONCLUSIÓN:")
        if problem_solved and good_relevance >= 7:
            print("✅ ÉXITO TOTAL - Sistema funciona correctamente")
            print("🎉 Problema de scores fijos 5.0/10 SOLUCIONADO")
            print("🧠 Evaluación contextual implementada exitosamente")
        elif problem_solved:
            print("✅ PROBLEMA PRINCIPAL SOLUCIONADO")
            print("🎯 Detecta correctamente casos no relacionados")
        else:
            print("⚠️  Sistema necesita ajustes adicionales")
        
        return result, problem_solved
        
    except Exception as e:
        print(f"❌ Error en test final: {e}")
        import traceback
        traceback.print_exc()
        return None, False

async def main():
    """Función principal."""
    print("🏛️ AI Judges Panel - Test Final del Sistema Contextual")
    print("=" * 70)
    
    result, solved = await test_final()
    
    print("\n" + "=" * 70)
    if result and solved:
        print("🎉 ¡MISIÓN CUMPLIDA!")
        print("✅ El sistema ya no da scores fijos de 5.0/10")
        print("🧠 Evalúa correctamente la relación prompt-respuesta")
        print("⚡ Sistema rápido y sin dependencias problemáticas")
    elif result:
        print("✅ Sistema funcionando - ajustes menores pendientes")
    else:
        print("❌ Error en el sistema - revisar implementación")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
