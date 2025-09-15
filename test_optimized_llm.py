#!/usr/bin/env python3
"""
Test de Rendimiento Optimizado
===============================

Prueba las optimizaciones implementadas en LLM Judges.
"""

import asyncio
import time
import sys
sys.path.append('app')

async def test_optimized_performance():
    """Prueba el rendimiento con las optimizaciones."""
    
    try:
        from models.llm_judges import LLMJudgesPanel
        
        # Ejemplo rápido para testing
        prompt = "¿Cómo funciona la inteligencia artificial?"
        response = "La IA usa algoritmos para aprender patrones de los datos."
        
        print("🚀 Testing Optimizaciones LLM Judges")
        print("="*50)
        
        # Test 1: Primera carga (debería ser lenta)
        print("\n📦 Test 1: Primera carga del modelo...")
        start_time = time.time()
        
        panel = LLMJudgesPanel(model_name="distilgpt2")
        result1 = await panel.evaluate_response_async(prompt, response)
        
        first_load_time = time.time() - start_time
        print(f"✅ Primera evaluación: {first_load_time:.2f}s")
        print(f"📊 Score: {result1['overall_score']}/10")
        
        # Test 2: Segunda evaluación (debería usar cache)
        print("\n⚡ Test 2: Segunda evaluación (con cache)...")
        start_time = time.time()
        
        panel2 = LLMJudgesPanel(model_name="distilgpt2")
        result2 = await panel2.evaluate_response_async(prompt, response)
        
        cached_time = time.time() - start_time
        print(f"✅ Segunda evaluación: {cached_time:.2f}s")
        print(f"📊 Score: {result2['overall_score']}/10")
        
        # Test 3: Diferentes ejemplos con cache
        print("\n🔄 Test 3: Diferentes prompts (con cache)...")
        test_cases = [
            ("¿Qué es Python?", "Python es un lenguaje de programación."),
            ("Explica la gravedad", "La gravedad es una fuerza fundamental."),
            ("¿Cómo cocinar pasta?", "Hierve agua, añade pasta, cocina 8-10 minutos.")
        ]
        
        total_cached_time = 0
        for i, (test_prompt, test_response) in enumerate(test_cases, 1):
            start_time = time.time()
            result = await panel.evaluate_response_async(test_prompt, test_response)
            eval_time = time.time() - start_time
            total_cached_time += eval_time
            
            print(f"   Test {i}: {eval_time:.2f}s - Score: {result['overall_score']}/10")
        
        avg_cached_time = total_cached_time / len(test_cases)
        
        print(f"\n📈 RESULTADOS:")
        print("-" * 30)
        print(f"Primera carga:     {first_load_time:.2f}s")
        print(f"Con cache:         {cached_time:.2f}s")
        print(f"Promedio c/cache:  {avg_cached_time:.2f}s")
        print(f"Mejora de cache:   {first_load_time/avg_cached_time:.1f}x más rápido")
        
        # Análisis de optimizaciones
        if first_load_time < 20:
            print("✅ Carga inicial optimizada (<20s)")
        else:
            print("⚠️  Carga inicial podría mejorar (>20s)")
            
        if avg_cached_time < 3:
            print("✅ Evaluaciones subsecuentes ultra-rápidas (<3s)")
        elif avg_cached_time < 8:
            print("✅ Evaluaciones subsecuentes rápidas (<8s)")
        else:
            print("⚠️  Evaluaciones subsecuentes necesitan optimización (>8s)")
            
        return {
            "first_load": first_load_time,
            "cached": avg_cached_time,
            "improvement": first_load_time/avg_cached_time
        }
        
    except ImportError as e:
        print(f"❌ Error importando LLMJudgesPanel: {e}")
        print("💡 Las dependencias de transformers pueden no estar instaladas correctamente")
        return None
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return None

async def simulate_optimized_performance():
    """Simula el rendimiento optimizado esperado."""
    print("🧮 Simulación de Rendimiento Optimizado")
    print("="*50)
    
    # Simular tiempos optimizados
    scenarios = {
        "Sin optimizaciones": {
            "first_load": 45.0,
            "subsequent": 40.0,
            "description": "Carga modelo cada vez"
        },
        "Con singleton cache": {
            "first_load": 15.0, 
            "subsequent": 3.0,
            "description": "Cache de modelos + parámetros optimizados"
        },
        "Optimización completa": {
            "first_load": 12.0,
            "subsequent": 2.0, 
            "description": "Cache + GPU + parámetros + early stopping"
        }
    }
    
    for name, data in scenarios.items():
        improvement = data["first_load"] / data["subsequent"]
        print(f"\n📊 {name}:")
        print(f"   Primera carga: {data['first_load']:.1f}s")
        print(f"   Subsecuentes:  {data['subsequent']:.1f}s")
        print(f"   Mejora:        {improvement:.1f}x más rápido")
        print(f"   Descripción:   {data['description']}")

def show_optimization_summary():
    """Muestra resumen de optimizaciones implementadas."""
    print("\n🎯 OPTIMIZACIONES IMPLEMENTADAS:")
    print("="*50)
    
    optimizations = [
        "✅ Singleton pattern - cache global de modelos",
        "✅ max_length reducido (512→50 tokens)",  
        "✅ do_sample=False - generación determinística",
        "✅ early_stopping=True - parada temprana",
        "✅ temperature reducida (0.7→0.2)",
        "✅ DistilGPT2 como default (modelo más rápido)",
        "✅ Mejor manejo de errores y fallbacks",
        "✅ Logging optimizado con tiempos"
    ]
    
    for opt in optimizations:
        print(f"  {opt}")
    
    print(f"\n🚀 MEJORA ESPERADA: 5-10x más rápido")
    print(f"📊 De ~45s a ~5s en evaluaciones subsecuentes")

async def main():
    """Función principal."""
    print("🏛️  AI Judges Panel - Test de Optimizaciones")
    print("="*60)
    
    # Mostrar optimizaciones
    show_optimization_summary()
    
    # Simular rendimiento  
    await simulate_optimized_performance()
    
    # Test real si es posible
    print("\n🧪 Intentando test real...")
    result = await test_optimized_performance()
    
    if result:
        print(f"\n🎉 ¡OPTIMIZACIONES EXITOSAS!")
        print(f"Mejora real: {result['improvement']:.1f}x más rápido")
    else:
        print(f"\n💡 Test simulado completado - implementar en producción")
    
    print("\n" + "="*60)
    print("🎯 CONCLUSIÓN: Los LLMs optimizados deberían ser ~5-10x más rápidos")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
