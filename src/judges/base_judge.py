"""
Base Judge Class for AI Evaluation System
=========================================

Esta clase base define la interfaz común que todos los jueces especializados
deben implementar para evaluar respuestas de LLMs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
import time
from enum import Enum


class JudgmentAspect(Enum):
    """Aspectos que pueden evaluar los jueces"""
    PRECISION = "precision"
    CREATIVITY = "creativity" 
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    EFFICIENCY = "efficiency"


@dataclass
class JudgeEvaluation:
    """Resultado de evaluación de un juez individual"""
    aspect: JudgmentAspect
    score: float  # 0-10
    reasoning: str
    strengths: List[str]
    improvements: List[str]
    confidence: float  # 0-1
    evaluation_time: float
    metadata: Dict[str, Any]


@dataclass 
class EvaluationContext:
    """Contexto para la evaluación"""
    original_prompt: str
    candidate_response: str
    reference_response: Optional[str] = None
    domain: Optional[str] = None
    task_type: Optional[str] = None
    evaluation_criteria: Optional[Dict[str, Any]] = None


class BaseJudge(ABC):
    """
    Clase base abstracta para todos los jueces evaluadores.
    
    Cada juez especializado debe heredar de esta clase e implementar
    los métodos abstractos para proporcionar evaluaciones específicas
    de su área de expertise.
    """
    
    def __init__(self, 
                 name: str,
                 specialty: JudgmentAspect,
                 model_name: str = "gpt-4",
                 temperature: float = 0.3):
        """
        Inicializa el juez base.
        
        Args:
            name: Nombre descriptivo del juez
            specialty: Aspecto que evalúa este juez  
            model_name: Modelo LLM a usar para evaluación
            temperature: Temperatura para generación (0-1)
        """
        self.name = name
        self.specialty = specialty
        self.model_name = model_name
        self.temperature = temperature
        self._evaluation_history: List[JudgeEvaluation] = []
        
    @abstractmethod
    def _generate_system_prompt(self) -> str:
        """
        Genera el prompt de sistema específico para este juez.
        
        Returns:
            String con el prompt de sistema que define la personalidad
            y expertise del juez
        """
        pass
        
    @abstractmethod 
    def _generate_evaluation_prompt(self, context: EvaluationContext) -> str:
        """
        Genera el prompt específico para evaluar una respuesta.
        
        Args:
            context: Contexto de evaluación con prompt original y respuesta
            
        Returns:
            String con el prompt de evaluación específico
        """
        pass
        
    @abstractmethod
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parsea la respuesta del LLM a un formato estructurado.
        
        Args:
            llm_response: Respuesta cruda del LLM
            
        Returns:
            Diccionario con score, reasoning, strengths, improvements
        """
        pass
        
    def evaluate(self, context: EvaluationContext) -> JudgeEvaluation:
        """
        Evalúa una respuesta usando este juez especializado.
        
        Args:
            context: Contexto de evaluación
            
        Returns:
            Evaluación completa del juez
        """
        start_time = time.time()
        
        try:
            # 1. Generar prompts
            system_prompt = self._generate_system_prompt()
            evaluation_prompt = self._generate_evaluation_prompt(context)
            
            # 2. Llamar al LLM (simulado por ahora)
            llm_response = self._call_llm(system_prompt, evaluation_prompt)
            
            # 3. Parsear respuesta
            parsed_result = self._parse_llm_response(llm_response)
            
            # 4. Crear evaluación
            evaluation = JudgeEvaluation(
                aspect=self.specialty,
                score=parsed_result["score"],
                reasoning=parsed_result["reasoning"],
                strengths=parsed_result["strengths"],
                improvements=parsed_result["improvements"],
                confidence=parsed_result.get("confidence", 0.8),
                evaluation_time=time.time() - start_time,
                metadata={
                    "model_used": self.model_name,
                    "temperature": self.temperature,
                    "judge_name": self.name
                }
            )
            
            # 5. Guardar en historial
            self._evaluation_history.append(evaluation)
            
            return evaluation
            
        except Exception as e:
            # Evaluación de fallback en caso de error
            return self._create_fallback_evaluation(str(e), time.time() - start_time)
            
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Llama al LLM para obtener la evaluación.
        
        Por ahora simula la llamada. En implementación real,
        aquí se haría la llamada API al modelo correspondiente.
        
        Args:
            system_prompt: Prompt de sistema
            user_prompt: Prompt de usuario
            
        Returns:
            Respuesta simulada del LLM
        """
        # NOTA: Esta es una implementación simulada
        # En la versión real, aquí iría la llamada a OpenAI, Anthropic, etc.
        
        simulated_response = {
            "score": 7.5,
            "reasoning": f"Como {self.name}, evalúo esta respuesta considerando {self.specialty.value}. La respuesta muestra calidad general pero tiene áreas de mejora.",
            "strengths": ["Claridad en la exposición", "Estructura lógica básica"],
            "improvements": ["Mayor profundidad en el análisis", "Más ejemplos específicos"],
            "confidence": 0.85
        }
        
        return json.dumps(simulated_response)
        
    def _create_fallback_evaluation(self, error: str, elapsed_time: float) -> JudgeEvaluation:
        """
        Crea una evaluación de fallback cuando ocurre un error.
        
        Args:
            error: Descripción del error
            elapsed_time: Tiempo transcurrido
            
        Returns:
            Evaluación de fallback
        """
        return JudgeEvaluation(
            aspect=self.specialty,
            score=5.0,  # Puntaje neutro
            reasoning=f"Error en evaluación de {self.name}: {error}",
            strengths=["Evaluación no completada"],
            improvements=["Resolver error técnico"],
            confidence=0.0,
            evaluation_time=elapsed_time,
            metadata={
                "error": True,
                "error_message": error,
                "judge_name": self.name
            }
        )
        
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de las evaluaciones realizadas.
        
        Returns:
            Diccionario con estadísticas del juez
        """
        if not self._evaluation_history:
            return {"total_evaluations": 0}
            
        scores = [eval.score for eval in self._evaluation_history]
        times = [eval.evaluation_time for eval in self._evaluation_history]
        
        return {
            "total_evaluations": len(self._evaluation_history),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "average_time": sum(times) / len(times),
            "specialty": self.specialty.value,
            "judge_name": self.name
        }
        
    def __str__(self) -> str:
        """Representación string del juez"""
        return f"{self.name} (Especialidad: {self.specialty.value})"
        
    def __repr__(self) -> str:
        """Representación detallada del juez"""
        return f"BaseJudge(name='{self.name}', specialty='{self.specialty.value}', model='{self.model_name}')"
