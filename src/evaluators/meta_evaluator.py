"""
Meta-Evaluator: Orquestador del Panel de Jueces
===============================================

Este evaluador coordina múltiples jueces especializados y combina
sus evaluaciones en un resultado final comprehensivo.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import statistics
import time
import json

# Importar las clases base
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judges.base_judge import BaseJudge, JudgeEvaluation, EvaluationContext, JudgmentAspect
from judges.precision_judge import PrecisionJudge


@dataclass
class ComprehensiveEvaluation:
    """Resultado de evaluación comprehensiva del panel"""
    final_score: float
    individual_scores: Dict[str, float]
    consensus_level: float  # 0-1, qué tan de acuerdo están los jueces
    strengths: List[str]
    improvements: List[str]
    detailed_feedback: Dict[str, JudgeEvaluation]
    evaluation_time: float
    metadata: Dict[str, Any]


class MetaEvaluator:
    """
    Meta-Evaluador que coordina el panel de jueces especializados.
    
    Responsabilidades:
    1. Coordinar evaluaciones de múltiples jueces
    2. Detectar consensos y discrepancias
    3. Combinar scores con pesos personalizables
    4. Generar reportes comprehensivos
    5. Proporcionar insights sobre calidad multi-dimensional
    """
    
    def __init__(self, 
                 judges: Optional[List[BaseJudge]] = None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Inicializa el Meta-Evaluador.
        
        Args:
            judges: Lista de jueces especializados. Si es None, usa panel por defecto
            weights: Pesos para cada aspecto. Si es None, usa pesos balanceados
        """
        # Panel de jueces por defecto
        if judges is None:
            self.judges = self._create_default_judges()
        else:
            self.judges = judges
            
        # Pesos por defecto balanceados
        self.weights = weights or {
            "precision": 0.25,
            "creativity": 0.20,
            "coherence": 0.25,
            "relevance": 0.20,
            "efficiency": 0.10
        }
        
        # Normalizar pesos para que sumen 1.0
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
        self._evaluation_history: List[ComprehensiveEvaluation] = []
        
    def _create_default_judges(self) -> List[BaseJudge]:
        """
        Crea el panel de jueces por defecto.
        
        Returns:
            Lista con todos los jueces especializados
        """
        judges = []
        
        # Juez de Precisión (ya implementado)
        judges.append(PrecisionJudge())
        
        # Placeholder para otros jueces (se implementarán después)
        # judges.append(CreativityJudge())
        # judges.append(CoherenceJudge()) 
        # judges.append(RelevanceJudge())
        # judges.append(EfficiencyJudge())
        
        # Por ahora, crear jueces simulados para demostrar la arquitectura
        judges.extend(self._create_simulated_judges())
        
        return judges
        
    def _create_simulated_judges(self) -> List[BaseJudge]:
        """
        Crea jueces simulados para demostrar la funcionalidad completa.
        
        Returns:
            Lista de jueces simulados
        """
        simulated_judges = []
        
        # Estos serían reemplazados por implementaciones reales
        judge_configs = [
            ("Dra. Creatividad", JudgmentAspect.CREATIVITY),
            ("Prof. Coherencia", JudgmentAspect.COHERENCE),
            ("Lic. Relevancia", JudgmentAspect.RELEVANCE),
            ("Ed. Eficiencia", JudgmentAspect.EFFICIENCY)
        ]
        
        for name, specialty in judge_configs:
            judge = self._create_simulated_judge(name, specialty)
            simulated_judges.append(judge)
            
        return simulated_judges
        
    def _create_simulated_judge(self, name: str, specialty: JudgmentAspect) -> BaseJudge:
        """
        Crea un juez simulado para completar el panel.
        
        Args:
            name: Nombre del juez
            specialty: Especialidad del juez
            
        Returns:
            Juez simulado que implementa BaseJudge
        """
        class SimulatedJudge(BaseJudge):
            def __init__(self, judge_name: str, judge_specialty: JudgmentAspect):
                super().__init__(judge_name, judge_specialty)
                
            def _generate_system_prompt(self) -> str:
                return f"Eres {self.name}, experto en {self.specialty.value}."
                
            def _generate_evaluation_prompt(self, context: EvaluationContext) -> str:
                return f"Evalúa esta respuesta enfocándote en {self.specialty.value}:\n\n{context.candidate_response}"
                
            def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
                try:
                    return json.loads(llm_response)
                except:
                    return {
                        "score": 7.0,
                        "reasoning": f"Evaluación simulada de {self.specialty.value}",
                        "strengths": ["Evaluación simulada"],
                        "improvements": ["Implementar evaluación real"],
                        "confidence": 0.7
                    }
                    
            def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
                # Simular variabilidad en scores según especialidad
                import random
                base_score = {
                    JudgmentAspect.CREATIVITY: 7.5,
                    JudgmentAspect.COHERENCE: 8.0,
                    JudgmentAspect.RELEVANCE: 7.8,
                    JudgmentAspect.EFFICIENCY: 6.5
                }.get(self.specialty, 7.0)
                
                score = base_score + random.uniform(-1.5, 1.5)
                score = max(0, min(10, score))
                
                response = {
                    "score": round(score, 1),
                    "reasoning": f"Evaluación desde la perspectiva de {self.specialty.value}. La respuesta muestra características apropiadas para este aspecto.",
                    "strengths": [f"Buen manejo de {self.specialty.value}", "Enfoque apropiado"],
                    "improvements": [f"Mejorar algunos aspectos de {self.specialty.value}", "Mayor desarrollo"],
                    "confidence": 0.75
                }
                
                return json.dumps(response)
                
        return SimulatedJudge(name, specialty)
        
    def evaluate(self, 
                 prompt: str, 
                 response: str,
                 reference_response: Optional[str] = None,
                 domain: Optional[str] = None,
                 include_automatic_metrics: bool = True) -> ComprehensiveEvaluation:
        """
        Evalúa una respuesta usando todo el panel de jueces.
        
        Args:
            prompt: Pregunta o prompt original
            response: Respuesta del candidato a evaluar
            reference_response: Respuesta de referencia opcional
            domain: Dominio específico para la evaluación
            include_automatic_metrics: Si incluir métricas automáticas (BLEU, ROUGE, etc.)
            
        Returns:
            Evaluación comprehensiva del panel completo
        """
        start_time = time.time()
        
        # Crear contexto de evaluación
        context = EvaluationContext(
            original_prompt=prompt,
            candidate_response=response,
            reference_response=reference_response,
            domain=domain
        )
        
        # Evaluar con cada juez
        individual_evaluations = {}
        individual_scores = {}
        
        for judge in self.judges:
            try:
                evaluation = judge.evaluate(context)
                aspect_name = evaluation.aspect.value
                individual_evaluations[aspect_name] = evaluation
                individual_scores[aspect_name] = evaluation.score
            except Exception as e:
                # Manejar errores de jueces individuales
                print(f"Error en {judge.name}: {e}")
                aspect_name = judge.specialty.value
                individual_scores[aspect_name] = 5.0  # Score neutro por error
                
        # Calcular score final ponderado
        final_score = self._calculate_weighted_score(individual_scores)
        
        # Analizar consenso entre jueces
        consensus_level = self._calculate_consensus(individual_scores)
        
        # Combinar strengths e improvements
        all_strengths, all_improvements = self._combine_feedback(individual_evaluations)
        
        # Métricas automáticas opcionales
        automatic_metrics = {}
        if include_automatic_metrics and reference_response:
            automatic_metrics = self._calculate_automatic_metrics(response, reference_response)
            
        # Crear evaluación comprehensiva
        comprehensive_eval = ComprehensiveEvaluation(
            final_score=final_score,
            individual_scores=individual_scores,
            consensus_level=consensus_level,
            strengths=all_strengths,
            improvements=all_improvements,
            detailed_feedback=individual_evaluations,
            evaluation_time=time.time() - start_time,
            metadata={
                "judges_count": len(self.judges),
                "weights_used": self.weights.copy(),
                "domain": domain,
                "automatic_metrics": automatic_metrics,
                "has_reference": reference_response is not None
            }
        )
        
        # Guardar en historial
        self._evaluation_history.append(comprehensive_eval)
        
        return comprehensive_eval
        
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        Calcula el score final usando pesos personalizados.
        
        Args:
            scores: Diccionario con scores individuales
            
        Returns:
            Score ponderado final (0-10)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for aspect, score in scores.items():
            weight = self.weights.get(aspect, 0.2)  # Peso por defecto
            weighted_sum += score * weight
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 5.0
        
    def _calculate_consensus(self, scores: Dict[str, float]) -> float:
        """
        Calcula el nivel de consenso entre jueces (0-1).
        
        Args:
            scores: Diccionario con scores individuales
            
        Returns:
            Nivel de consenso (0 = mucha discrepancia, 1 = consenso total)
        """
        if len(scores) < 2:
            return 1.0
            
        score_values = list(scores.values())
        std_dev = statistics.stdev(score_values)
        
        # Normalizar: std_dev de 0 = consenso perfecto, std_dev de 5 = máxima discrepancia
        consensus = max(0.0, 1.0 - (std_dev / 5.0))
        return round(consensus, 3)
        
    def _combine_feedback(self, evaluations: Dict[str, JudgeEvaluation]) -> Tuple[List[str], List[str]]:
        """
        Combina strengths e improvements de todos los jueces.
        
        Args:
            evaluations: Evaluaciones individuales de cada juez
            
        Returns:
            Tupla con (strengths combinados, improvements combinados)
        """
        all_strengths = []
        all_improvements = []
        
        for aspect, evaluation in evaluations.items():
            # Agregar contexto del juez a cada strength/improvement
            for strength in evaluation.strengths:
                all_strengths.append(f"[{aspect.title()}] {strength}")
                
            for improvement in evaluation.improvements:
                all_improvements.append(f"[{aspect.title()}] {improvement}")
                
        return all_strengths, all_improvements
        
    def _calculate_automatic_metrics(self, response: str, reference: str) -> Dict[str, Any]:
        """
        Calcula métricas automáticas complementarias.
        
        Args:
            response: Respuesta del candidato
            reference: Respuesta de referencia
            
        Returns:
            Diccionario con métricas automáticas
        """
        # Placeholder para métricas automáticas reales
        # En implementación completa, aquí se calcularían BLEU, ROUGE, BERTScore, etc.
        
        # Simulación simple por ahora
        import random
        
        return {
            "bleu_score": round(random.uniform(0.3, 0.8), 3),
            "rouge_1": round(random.uniform(0.4, 0.9), 3),
            "rouge_l": round(random.uniform(0.3, 0.8), 3),
            "bert_score": round(random.uniform(0.6, 0.95), 3),
            "length_ratio": len(response) / len(reference) if reference else 1.0
        }
        
    def get_panel_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del desempeño del panel de jueces.
        
        Returns:
            Resumen estadístico del panel
        """
        if not self._evaluation_history:
            return {"error": "No hay evaluaciones para resumir"}
            
        total_evaluations = len(self._evaluation_history)
        avg_final_score = statistics.mean([e.final_score for e in self._evaluation_history])
        avg_consensus = statistics.mean([e.consensus_level for e in self._evaluation_history])
        avg_time = statistics.mean([e.evaluation_time for e in self._evaluation_history])
        
        # Análisis por aspecto
        aspect_performance = {}
        for eval in self._evaluation_history:
            for aspect, score in eval.individual_scores.items():
                if aspect not in aspect_performance:
                    aspect_performance[aspect] = []
                aspect_performance[aspect].append(score)
                
        aspect_averages = {
            aspect: statistics.mean(scores) 
            for aspect, scores in aspect_performance.items()
        }
        
        return {
            "total_evaluations": total_evaluations,
            "average_final_score": round(avg_final_score, 2),
            "average_consensus": round(avg_consensus, 3),
            "average_evaluation_time": round(avg_time, 2),
            "aspect_performance": {k: round(v, 2) for k, v in aspect_averages.items()},
            "judges_count": len(self.judges),
            "weights_distribution": self.weights
        }
        
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Actualiza los pesos del panel para futuros evaluaciones.
        
        Args:
            new_weights: Nuevos pesos por aspecto
        """
        total = sum(new_weights.values())
        self.weights = {k: v/total for k, v in new_weights.items()}
        
    def add_judge(self, judge: BaseJudge):
        """
        Añade un nuevo juez al panel.
        
        Args:
            judge: Nuevo juez especializado
        """
        self.judges.append(judge)
        
    def remove_judge(self, judge_name: str) -> bool:
        """
        Remueve un juez del panel.
        
        Args:
            judge_name: Nombre del juez a remover
            
        Returns:
            True si se removió exitosamente
        """
        for i, judge in enumerate(self.judges):
            if judge.name == judge_name:
                del self.judges[i]
                return True
        return False
        
    def __str__(self) -> str:
        """Representación string del meta-evaluador"""
        judges_names = [judge.name for judge in self.judges]
        return f"MetaEvaluator with {len(self.judges)} judges: {', '.join(judges_names)}"
