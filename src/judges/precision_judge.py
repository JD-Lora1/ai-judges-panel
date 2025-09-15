"""
Precision Judge - Especialista en Factualidad y Exactitud
========================================================

Este juez evalúa la precisión factual, exactitud de la información
y ausencia de alucinaciones en las respuestas de LLMs.
"""

import json
import re
from typing import Dict, Any
from .base_judge import BaseJudge, JudgmentAspect, EvaluationContext


class PrecisionJudge(BaseJudge):
    """
    Juez especializado en evaluar precisión y factualidad.
    
    Se enfoca en:
    - Veracidad de hechos presentados
    - Ausencia de alucinaciones
    - Precisión en datos numéricos y fechas
    - Correctitud de citas y referencias
    - Consistencia interna de la información
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.2):
        """
        Inicializa el Juez de Precisión.
        
        Args:
            model_name: Modelo LLM para evaluación
            temperature: Temperatura baja para evaluaciones consistentes
        """
        super().__init__(
            name="Dr. Precisión", 
            specialty=JudgmentAspect.PRECISION,
            model_name=model_name,
            temperature=temperature
        )
        
    def _generate_system_prompt(self) -> str:
        """
        Genera el prompt de sistema que define la personalidad del juez.
        
        Returns:
            Prompt de sistema especializado en verificación de hechos
        """
        return """Eres el Dr. Precisión, un experto verificador de hechos con 20 años de experiencia en periodismo investigativo y fact-checking. 

Tu especialidad es detectar:
- Información incorrecta o inexacta
- Alucinaciones y fabricación de datos
- Inconsistencias internas en el texto
- Fechas, números o estadísticas erróneas
- Referencias inexistentes o incorrectas
- Afirmaciones sin fundamento

Evaluación rigurosa pero justa:
- Eres minucioso pero no pedante
- Reconoces cuando la información es verificable vs opiniones legítimas
- Distingues entre errores menores y fallas graves de factualidad
- Proporcionas feedback específico y constructivo

Tu objetivo: Asegurar que la IA produzca información confiable y verificable."""
        
    def _generate_evaluation_prompt(self, context: EvaluationContext) -> str:
        """
        Genera el prompt específico para evaluar precisión.
        
        Args:
            context: Contexto con el prompt original y respuesta del candidato
            
        Returns:
            Prompt de evaluación especializado en precisión
        """
        return f"""
PREGUNTA ORIGINAL:
{context.original_prompt}

RESPUESTA A EVALUAR:
{context.candidate_response}

INSTRUCCIONES DE EVALUACIÓN:

Como Dr. Precisión, evalúa esta respuesta enfocándote EXCLUSIVAMENTE en PRECISIÓN y FACTUALIDAD:

1. **VERIFICACIÓN DE HECHOS** (40% peso):
   - ¿Los hechos presentados son correctos?
   - ¿Hay fechas, números o datos verificables?
   - ¿Las referencias mencionadas existen realmente?

2. **DETECCIÓN DE ALUCINACIONES** (30% peso):
   - ¿Se inventó nombres, lugares, estudios o citas?
   - ¿Hay afirmaciones específicas sin base real?
   - ¿Se presentan como hechos cosas que son especulación?

3. **CONSISTENCIA INTERNA** (20% peso):
   - ¿La respuesta es internamente consistente?
   - ¿Hay contradicciones en la información presentada?
   - ¿Los números y estadísticas son coherentes entre sí?

4. **CALIDAD DE EVIDENCIA** (10% peso):
   - ¿Se distingue claramente entre hechos y opiniones?
   - ¿Se indican las limitaciones del conocimiento?
   - ¿Se evitan afirmaciones absolutas sin fundamento?

FORMATO DE RESPUESTA REQUERIDO (JSON):
{{
    "score": [número 0-10],
    "reasoning": "Análisis detallado de la precisión factual...",
    "strengths": ["Fortaleza 1", "Fortaleza 2", "..."],
    "improvements": ["Mejora 1", "Mejora 2", "..."],
    "confidence": [número 0.0-1.0],
    "factual_claims": ["Lista de afirmaciones fácticas encontradas"],
    "potential_hallucinations": ["Lista de posibles alucinaciones"],
    "verification_notes": "Notas sobre verificabilidad"
}}

CRITERIOS DE PUNTUACIÓN:
- 9-10: Información altamente precisa y verificable
- 7-8: Mayormente preciso con errores menores
- 5-6: Precisión mixta, algunos errores significativos
- 3-4: Varios errores o información dudosa
- 0-2: Información incorrecta o altamente no confiable

Sé riguroso pero justo. Solo evalúa PRECISIÓN, no otros aspectos como creatividad o estilo.
"""
        
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parsea la respuesta del LLM especializado en precisión.
        
        Args:
            llm_response: Respuesta JSON del LLM
            
        Returns:
            Diccionario estructurado con la evaluación
        """
        try:
            # Intentar parsear JSON directamente
            data = json.loads(llm_response)
            
            # Validar campos obligatorios
            required_fields = ["score", "reasoning", "strengths", "improvements", "confidence"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Campo requerido faltante: {field}")
                    
            # Normalizar score a rango 0-10
            score = float(data["score"])
            score = max(0.0, min(10.0, score))
            
            # Normalizar confidence a rango 0-1
            confidence = float(data.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                "score": score,
                "reasoning": str(data["reasoning"]),
                "strengths": list(data["strengths"]),
                "improvements": list(data["improvements"]),
                "confidence": confidence,
                # Campos específicos del Juez de Precisión
                "factual_claims": data.get("factual_claims", []),
                "potential_hallucinations": data.get("potential_hallucinations", []),
                "verification_notes": data.get("verification_notes", "")
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: intenta extraer información con regex
            return self._extract_with_regex(llm_response)
            
    def _extract_with_regex(self, llm_response: str) -> Dict[str, Any]:
        """
        Extrae información usando regex cuando el JSON falla.
        
        Args:
            llm_response: Respuesta cruda del LLM
            
        Returns:
            Diccionario con información extraída
        """
        # Buscar score con regex
        score_match = re.search(r'score["\']?\s*:\s*([0-9.]+)', llm_response, re.IGNORECASE)
        score = 5.0  # Default neutro
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(10.0, score))
            except ValueError:
                pass
                
        # Buscar reasoning
        reasoning_match = re.search(r'reasoning["\']?\s*:\s*["\']([^"\']+)["\']', llm_response, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1) if reasoning_match else "Evaluación de precisión completada con parsing parcial."
        
        # Buscar listas (simplificado)
        strengths = self._extract_list(llm_response, "strengths") or ["Información procesada"]
        improvements = self._extract_list(llm_response, "improvements") or ["Mejorar verificabilidad"]
        
        return {
            "score": score,
            "reasoning": reasoning,
            "strengths": strengths,
            "improvements": improvements,
            "confidence": 0.6,  # Confidence baja por parsing parcial
            "factual_claims": [],
            "potential_hallucinations": [],
            "verification_notes": "Información extraída con parsing de emergencia"
        }
        
    def _extract_list(self, text: str, list_name: str) -> list:
        """
        Extrae una lista usando regex.
        
        Args:
            text: Texto donde buscar
            list_name: Nombre de la lista a extraer
            
        Returns:
            Lista extraída o None si no se encuentra
        """
        pattern = rf'{list_name}["\']?\s*:\s*\[(.*?)\]'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            items_text = match.group(1)
            # Extraer elementos entre comillas
            items = re.findall(r'["\']([^"\']+)["\']', items_text)
            return items if items else []
        return []
        
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Simula una llamada al LLM específica para el Juez de Precisión.
        
        Args:
            system_prompt: Prompt de sistema
            user_prompt: Prompt de evaluación
            
        Returns:
            Respuesta JSON simulada del LLM
        """
        # Simulación más realista para el Juez de Precisión
        simulated_response = {
            "score": 8.2,
            "reasoning": "Como Dr. Precisión, he examinado cuidadosamente la respuesta. Los hechos presentados son en su mayoría correctos y verificables. Se observa una buena distinción entre hechos y opiniones. Hay una fecha que requiere verificación adicional, pero no se detectan alucinaciones significativas.",
            "strengths": [
                "Hechos principales correctos y verificables",
                "Distinción clara entre hechos y opiniones",
                "No se observan alucinaciones evidentes",
                "Información consistente internamente"
            ],
            "improvements": [
                "Verificar la fecha específica mencionada en el párrafo 2",
                "Agregar fuentes para las estadísticas presentadas",
                "Ser más específico sobre las limitaciones del conocimiento"
            ],
            "confidence": 0.87,
            "factual_claims": [
                "Einstein desarrolló la teoría de la relatividad",
                "E=mc² es una ecuación famosa de la física",
                "La velocidad de la luz es constante"
            ],
            "potential_hallucinations": [],
            "verification_notes": "Mayoría de afirmaciones son hechos científicos establecidos. Requiere verificación de fecha específica."
        }
        
        return json.dumps(simulated_response, ensure_ascii=False, indent=2)
        
    def get_precision_report(self) -> Dict[str, Any]:
        """
        Genera un reporte específico sobre las evaluaciones de precisión.
        
        Returns:
            Reporte detallado de tendencias en precisión
        """
        if not self._evaluation_history:
            return {"error": "No hay evaluaciones para reportar"}
            
        total_evaluations = len(self._evaluation_history)
        high_precision_count = sum(1 for eval in self._evaluation_history if eval.score >= 8.0)
        low_precision_count = sum(1 for eval in self._evaluation_history if eval.score < 5.0)
        
        avg_confidence = sum(eval.confidence for eval in self._evaluation_history) / total_evaluations
        
        return {
            "total_evaluations": total_evaluations,
            "high_precision_percentage": (high_precision_count / total_evaluations) * 100,
            "low_precision_percentage": (low_precision_count / total_evaluations) * 100,
            "average_confidence": avg_confidence,
            "judge_reliability": "Alta" if avg_confidence > 0.8 else "Media" if avg_confidence > 0.6 else "Baja",
            "specialty_focus": "Verificación de hechos y detección de alucinaciones"
        }
