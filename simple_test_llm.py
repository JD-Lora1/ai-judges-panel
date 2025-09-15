#!/usr/bin/env python3
"""
Prueba Simple de LLM Judges
============================

Prueba básica de la lógica de LLM judges sin dependencias externas complejas.
"""

import sys
import asyncio

# Simular la funcionalidad principal de LLM judges
class MockLLMJudgesPanel:
    """Mock del panel LLM para pruebas sin dependencias."""
    
    def __init__(self, model_name="google/flan-t5-base"):
        self.model_name = model_name
        self.model_loaded = False
        print(f"🧠 Mock LLM Panel inicializado con modelo: {model_name}")
    
    def _mock_llm_evaluation(self, prompt, response, criteria, judge_name):
        """Simula evaluación LLM basada en relación prompt-respuesta."""
        
        # Análisis básico de relevancia contextual
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        
        # Palabras clave del prompt problemático
        prompt_keywords = ["columnas", "domain", "metodologia", "fase", "tags", "filtro", "web", "card", "buscador"]
        response_keywords = ["design thinking", "agiles", "tracks", "backlogs", "equipos", "sprint", "foro"]
        
        # Contar coincidencias de contexto
        prompt_context_words = [word for word in prompt_keywords if word in prompt_lower]
        response_context_words = [word for word in response_keywords if word in response_lower]
        
        # Verificar si hay overlap semántico
        semantic_overlap = len(set(prompt_context_words) & set(response_context_words)) / max(len(prompt_context_words), 1)
        
        # Evaluación específica por juez
        if judge_name == "Lic. Relevancia":
            # Relevancia debería ser baja si no hay relación semántica
            base_score = 3.0 if semantic_overlap < 0.2 else 7.0
            feedback = f"Baja relevancia contextual - prompt habla de {', '.join(prompt_context_words[:3])} mientras respuesta habla de {', '.join(response_context_words[:3])}"
            
        elif judge_name == "Dr. Precisión":
            # Precisión baja si la respuesta no aborda lo solicitado
            addresses_request = any(word in response_lower for word in ["filtro", "columna", "buscador", "card"])
            base_score = 7.0 if addresses_request else 2.0
            feedback = "No responde a la solicitud específica sobre filtros web y columnas"
            
        elif judge_name == "Prof. Coherencia":
            # La respuesta es internamente coherente pero no con el prompt
            base_score = 7.5  # La respuesta sobre design thinking es coherente
            feedback = "Respuesta internamente coherente pero no relacionada con el prompt"
            
        elif judge_name == "Ed. Eficiencia":
            # Eficiencia baja porque no es útil para el usuario
            base_score = 3.0
            feedback = "Respuesta ineficiente - no ayuda con el problema planteado"
            
        else:  # Dra. Creatividad
            # Creatividad puede ser alta independientemente de la relevancia
            base_score = 6.0
            feedback = "Enfoque creativo pero mal dirigido"
        
        return {
            "score": base_score,
            "feedback": feedback,
            "semantic_overlap": semantic_overlap,
            "context_analysis": {
                "prompt_keywords": prompt_context_words,
                "response_keywords": response_context_words
            }
        }
    
    async def evaluate_response_async(self, prompt, response, weights=None):
        """Mock de evaluación asíncrona."""
        
        if weights is None:
            weights = {
                "precision": 0.35,
                "coherence": 0.30,
                "relevance": 0.20,
                "efficiency": 0.10,
                "creativity": 0.05
            }
        
        judges = [
            ("Dr. Precisión", "precision"),
            ("Prof. Coherencia", "coherence"),
            ("Lic. Relevancia", "relevance"),
            ("Ed. Eficiencia", "efficiency"),
            ("Dra. Creatividad", "creativity")
        ]
        
        individual_scores = {}
        scores_list = []
        
        for judge_name, aspect in judges:
            result = self._mock_llm_evaluation(prompt, response, f"Evaluar {aspect}", judge_name)
            individual_scores[aspect] = result
            scores_list.append(result["score"])
        
        # Calcular score final
        overall_score = sum(
            individual_scores[aspect]["score"] * weights[aspect] 
            for aspect in weights.keys()
        )
        
        # Calcular consenso
        mean_score = sum(scores_list) / len(scores_list)
        variance = sum((score - mean_score) ** 2 for score in scores_list) / len(scores_list)
        std_dev = variance ** 0.5
        consensus = max(0, 100 - (std_dev * 10))
        
        return {
            "overall_score": round(overall_score, 1),
            "consensus": round(consensus, 0),
            "model_used": f"Mock {self.model_name}",
            "individual_scores": individual_scores,
            "weights": weights,
            "evaluation_stats": {
                "mean_score": round(mean_score, 2),
                "std_deviation": round(std_dev, 2),
                "score_range": {"min": min(scores_list), "max": max(scores_list)}
            }
        }

def test_problematic_example():
    """El ejemplo problemático del usuario."""
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

    return prompt, response

def simulate_hf_judges():
    """Simula los resultados de HuggingFace judges (método actual)."""
    return {
        "overall_score": 7.0,
        "individual_scores": {
            "precision": 7.0,
            "coherence": 7.2,
            "relevance": 6.6,  # Probablemente alto por embeddings similares
            "efficiency": 7.0,
            "creativity": 8.5
        },
        "consensus": 87,
        "method": "HuggingFace Judges"
    }

async def main():
    print("🏛️  AI Judges Panel - Prueba de Evaluación Contextual")
    print("=" * 60)
    
    # Obtener ejemplo problemático
    prompt, response = test_problematic_example()
    
    print("\n🧪 EJEMPLO PROBLEMÁTICO:")
    print("📝 Prompt: Solicita ayuda con filtros, columnas y funcionalidades web")
    print("🤖 Respuesta: Habla sobre Design Thinking y metodologías ágiles")
    print("❌ PROBLEMA: No hay relación entre prompt y respuesta\n")
    
    # Simular HuggingFace judges
    print("🤖 EVALUACIÓN CON HUGGINGFACE JUDGES (método actual):")
    hf_result = simulate_hf_judges()
    print(f"   📊 Score general: {hf_result['overall_score']}/10")
    print(f"   🎯 Relevancia: {hf_result['individual_scores']['relevance']}/10")
    print("   ⚠️  PROBLEMA: Score relativamente alto a pesar de la falta de relación\n")
    
    # Probar mock LLM judges
    models = ["google/flan-t5-base", "microsoft/DialoGPT-medium", "distilgpt2"]
    
    for model in models:
        print(f"🧠 EVALUACIÓN CON LLM JUDGES - {model}:")
        llm_panel = MockLLMJudgesPanel(model_name=model)
        result = await llm_panel.evaluate_response_async(prompt, response)
        
        print(f"   📊 Score general: {result['overall_score']}/10")
        print(f"   🤝 Consenso: {result['consensus']}%")
        print("   📋 Análisis detallado:")
        
        for aspect, data in result['individual_scores'].items():
            print(f"      {aspect.capitalize()}: {data['score']}/10 - {data['feedback']}")
        
        print("")
    
    print("🔍 ANÁLISIS COMPARATIVO:")
    print("-" * 50)
    print("📊 Scores de Relevancia esperados:")
    print(f"   HF Judges: {hf_result['individual_scores']['relevance']}/10 (alto - problema)")
    print(f"   LLM Judges: ~3/10 (bajo - correcto)")
    print("\n🎯 CONCLUSIÓN:")
    print("   Los LLM Judges deberían detectar mejor la falta de relación")
    print("   contextual entre prompt y respuesta, dando scores más bajos.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
