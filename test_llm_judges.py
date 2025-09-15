#!/usr/bin/env python3
"""
Script de Prueba para LLM Judges
================================

Prueba el sistema de evaluación con LLM judges usando el ejemplo problemático
donde el prompt no tiene relación con la respuesta.
"""

import asyncio
import sys
sys.path.append('app')

from models.llm_judges import LLMJudgesPanel
from models.hf_judges import HuggingFaceJudgesPanel
import time

def test_problematic_example():
    """Prueba con el ejemplo problemático donde prompt y respuesta no tienen relación."""
    
    # El ejemplo problemático del usuario
    prompt = """chat, he agregado estas nuevas columnas: Domain	Metodologia	Fase	Tags
Para Domain, Metodologia, y Fase, quiero que me des un prompt para que en mi web se pueda hacer un filtro por esas nuevas columnas. Ademas en la parte inferior de la card, aparezca la Domain (Tipo) y la o las fases. Y que al hacer la busqueda en el buscador, tambien la haga por estas tags. Se consiczo y punutal, lenguaje natural"""

    response = """¡Perfecto! Entonces te dejo un **prompt listo** para que se lo des a otra IA y te redacte la participación:

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

    print("🧪 Iniciando prueba con ejemplo problemático...")
    print("\n" + "="*80)
    print("PROMPT:")
    print("-" * 40)
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    print("\n" + "-" * 40)
    print("RESPUESTA:")
    print("-" * 40)
    print(response[:200] + "..." if len(response) > 200 else response)
    print("="*80 + "\n")

    return prompt, response

async def test_hf_judges(prompt, response):
    """Prueba con HuggingFace judges (método original)."""
    print("🤖 Evaluando con HuggingFace Judges (método original)...")
    
    try:
        # Nota: Esta es una prueba simplificada ya que HuggingFaceJudgesPanel 
        # requiere una configuración más compleja
        print("⏱️  Simulando evaluación con HF Judges...")
        
        # Simulamos los scores que probablemente daría el sistema actual
        hf_results = {
            "overall_score": 7.0,
            "individual_scores": {
                "precision": 7.0,
                "coherence": 7.2,
                "relevance": 6.6,
                "efficiency": 7.0,
                "creativity": 8.5
            },
            "consensus": 87,
            "method": "HuggingFace Judges"
        }
        
        print("✅ HuggingFace Judges - Resultados:")
        print(f"   📊 Score general: {hf_results['overall_score']}/10")
        print(f"   🤝 Consenso: {hf_results['consensus']}%")
        print(f"   🎯 Relevancia: {hf_results['individual_scores']['relevance']}/10")
        print("   ⚠️  PROBLEMA: Score alto a pesar de que prompt y respuesta no están relacionados\n")
        
        return hf_results
        
    except Exception as e:
        print(f"❌ Error con HuggingFace Judges: {e}")
        return None

async def test_llm_judges_with_model(prompt, response, model_name):
    """Prueba con un modelo LLM específico."""
    print(f"🧠 Evaluando con LLM Judges - Modelo: {model_name}...")
    
    try:
        start_time = time.time()
        
        # Crear panel de jueces LLM
        llm_panel = LLMJudgesPanel(model_name=model_name)
        
        # Realizar evaluación
        result = await llm_panel.evaluate_response_async(prompt, response)
        
        eval_time = time.time() - start_time
        
        print(f"✅ LLM Judges ({result['model_used']}) - Resultados:")
        print(f"   📊 Score general: {result['overall_score']}/10")
        print(f"   🤝 Consenso: {result['consensus']}%")
        print(f"   ⏱️  Tiempo: {eval_time:.2f}s")
        print("   📋 Scores individuales:")
        for aspect, score_data in result['individual_scores'].items():
            print(f"      {aspect.capitalize()}: {score_data['score']}/10 - {score_data['feedback']}")
        print("")
        
        return result
        
    except Exception as e:
        print(f"❌ Error con LLM Judges ({model_name}): {e}")
        return None

async def main():
    """Función principal de prueba."""
    print("🏛️  AI Judges Panel - Prueba de Evaluación Contextual")
    print("=" * 60)
    
    # Obtener ejemplo problemático
    prompt, response = test_problematic_example()
    
    # Probar HuggingFace judges
    hf_result = await test_hf_judges(prompt, response)
    
    # Lista de modelos LLM para probar
    llm_models = [
        "google/flan-t5-base",
        "distilgpt2", 
        "microsoft/DialoGPT-small"
    ]
    
    llm_results = []
    
    # Probar cada modelo LLM
    for model in llm_models:
        result = await test_llm_judges_with_model(prompt, response, model)
        if result:
            llm_results.append(result)
        
        # Pausa entre evaluaciones para evitar sobrecarga
        await asyncio.sleep(1)
    
    # Comparación final
    print("🔍 ANÁLISIS COMPARATIVO:")
    print("-" * 50)
    
    if hf_result:
        print(f"HF Judges   - Relevancia: {hf_result['individual_scores']['relevance']}/10")
    
    for result in llm_results:
        relevance_score = result['individual_scores']['relevance']['score']
        model_name = result['model_used']
        print(f"LLM Judges ({model_name}) - Relevancia: {relevance_score}/10")
    
    print("\n" + "=" * 60)
    print("🎯 ESPERADO: Los LLM Judges deberían dar scores más bajos en relevancia")
    print("   ya que evalúan la relación contextual entre prompt y respuesta.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
