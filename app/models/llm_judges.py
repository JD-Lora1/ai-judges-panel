from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Dict, List, Optional, Tuple
import re
import logging
from dataclasses import dataclass
import asyncio
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    model_id: str
    max_length: int
    temperature: float = 0.7
    description: str = ""

class LLMJudgesPanel:
    """Panel de jueces que utiliza modelos de lenguaje reales para evaluación contextual."""
    
    AVAILABLE_MODELS = {
        "microsoft/DialoGPT-medium": ModelConfig(
            name="DialoGPT Medium",
            model_id="microsoft/DialoGPT-medium",
            max_length=512,
            temperature=0.3,
            description="Modelo conversacional de Microsoft, bueno para evaluar coherencia contextual"
        ),
        "google/flan-t5-base": ModelConfig(
            name="FLAN-T5 Base",
            model_id="google/flan-t5-base",
            max_length=512,
            temperature=0.5,
            description="Modelo instruction-following de Google, excelente para evaluación de relevancia"
        ),
        "distilbert/distilgpt2": ModelConfig(
            name="DistilGPT2",
            model_id="distilgpt2",
            max_length=512,
            temperature=0.4,
            description="Modelo ligero de generación de texto, rápido y eficiente"
        ),
        "microsoft/DialoGPT-small": ModelConfig(
            name="DialoGPT Small",
            model_id="microsoft/DialoGPT-small",
            max_length=256,
            temperature=0.3,
            description="Versión más ligera de DialoGPT, ideal para evaluaciones rápidas"
        )
    }
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.model_config = self.AVAILABLE_MODELS.get(model_name)
        if not self.model_config:
            raise ValueError(f"Modelo {model_name} no disponible. Modelos disponibles: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.pipeline = None
        self._load_model()
        
    def _load_model(self):
        """Carga el modelo seleccionado."""
        try:
            logger.info(f"Cargando modelo {self.model_config.name}...")
            
            if "flan-t5" in self.model_name:
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model_config.model_id,
                    max_length=self.model_config.max_length,
                    temperature=self.model_config.temperature,
                    do_sample=True
                )
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_config.model_id,
                    max_length=self.model_config.max_length,
                    temperature=self.model_config.temperature,
                    do_sample=True,
                    pad_token_id=50256
                )
            
            logger.info(f"Modelo {self.model_config.name} cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo {self.model_name}: {str(e)}")
            # Fallback a modelo más simple
            self.model_name = "distilgpt2"
            self.model_config = self.AVAILABLE_MODELS[self.model_name]
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_config.model_id,
                max_length=256,
                temperature=0.4,
                do_sample=True,
                pad_token_id=50256
            )
            logger.info("Fallback a DistilGPT2 completado")
    
    def _generate_evaluation_prompt(self, judge_name: str, prompt: str, response: str, criteria: str) -> str:
        """Genera el prompt para que el LLM evalúe como un juez específico."""
        
        base_prompt = f"""Actúa como {judge_name}, un experto evaluador de respuestas de IA.

Tu tarea es evaluar la calidad de una respuesta en relación a un prompt específico.

PROMPT ORIGINAL: "{prompt}"
RESPUESTA A EVALUAR: "{response}"

CRITERIOS DE EVALUACIÓN: {criteria}

Evalúa la respuesta en una escala del 1-10 considerando:
- ¿La respuesta aborda directamente lo solicitado en el prompt?
- ¿Es relevante y útil para quien hizo la pregunta?
- ¿Cumple con los criterios específicos?

Responde SOLO con un número del 1 al 10, seguido de una breve justificación (máximo 20 palabras).
Formato: "Puntuación: X/10. Justificación: [razón breve]"""
        
        return base_prompt
    
    def _extract_score_from_response(self, llm_response: str) -> Tuple[float, str]:
        """Extrae la puntuación y justificación de la respuesta del LLM."""
        try:
            # Buscar patrones como "Puntuación: 7/10" o "7/10" o "Score: 8"
            score_patterns = [
                r"Puntuación:\s*(\d+(?:\.\d+)?)/10",
                r"Score:\s*(\d+(?:\.\d+)?)/10",
                r"(\d+(?:\.\d+)?)/10",
                r"(\d+(?:\.\d+)?)\s*/\s*10",
                r"(\d+(?:\.\d+)?)"
            ]
            
            score = 5.0  # Default fallback
            for pattern in score_patterns:
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    extracted_score = float(match.group(1))
                    if 0 <= extracted_score <= 10:
                        score = extracted_score
                        break
            
            # Extraer justificación
            justification_match = re.search(r"Justificación:\s*(.+?)(?:\n|$)", llm_response, re.IGNORECASE)
            justification = justification_match.group(1).strip() if justification_match else "Evaluación automática"
            
            return score, justification
            
        except Exception as e:
            logger.error(f"Error extrayendo puntuación: {str(e)}")
            return 5.0, "Error en evaluación"
    
    def evaluate_precision(self, prompt: str, response: str) -> Dict:
        """Evalúa la precisión de la respuesta."""
        criteria = "Precisión factual, exactitud de la información, ausencia de errores, respuesta directa al prompt."
        eval_prompt = self._generate_evaluation_prompt("Dr. Precisión", prompt, response, criteria)
        
        try:
            if "flan-t5" in self.model_name:
                llm_response = self.pipeline(eval_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            else:
                llm_response = self.pipeline(eval_prompt, max_length=len(eval_prompt) + 100, num_return_sequences=1)[0]['generated_text']
                # Remover el prompt original de la respuesta
                llm_response = llm_response[len(eval_prompt):].strip()
            
            score, justification = self._extract_score_from_response(llm_response)
            
            return {
                "score": score,
                "feedback": f"[Precision] {justification}",
                "details": {
                    "raw_response": llm_response,
                    "model_used": self.model_config.name
                }
            }
        except Exception as e:
            logger.error(f"Error en evaluación de precisión: {str(e)}")
            return {
                "score": 5.0,
                "feedback": "[Precision] Error en evaluación automática",
                "details": {"error": str(e)}
            }
    
    def evaluate_coherence(self, prompt: str, response: str) -> Dict:
        """Evalúa la coherencia de la respuesta."""
        criteria = "Coherencia interna, fluidez narrativa, estructura lógica, conexión clara entre ideas."
        eval_prompt = self._generate_evaluation_prompt("Prof. Coherencia", prompt, response, criteria)
        
        try:
            if "flan-t5" in self.model_name:
                llm_response = self.pipeline(eval_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            else:
                llm_response = self.pipeline(eval_prompt, max_length=len(eval_prompt) + 100, num_return_sequences=1)[0]['generated_text']
                llm_response = llm_response[len(eval_prompt):].strip()
            
            score, justification = self._extract_score_from_response(llm_response)
            
            return {
                "score": score,
                "feedback": f"[Coherence] {justification}",
                "details": {
                    "raw_response": llm_response,
                    "model_used": self.model_config.name
                }
            }
        except Exception as e:
            logger.error(f"Error en evaluación de coherencia: {str(e)}")
            return {
                "score": 5.0,
                "feedback": "[Coherence] Error en evaluación automática",
                "details": {"error": str(e)}
            }
    
    def evaluate_relevance(self, prompt: str, response: str) -> Dict:
        """Evalúa la relevancia de la respuesta al prompt."""
        criteria = "Relevancia directa al prompt, pertinencia del contenido, enfoque en lo solicitado, utilidad para el usuario."
        eval_prompt = self._generate_evaluation_prompt("Lic. Relevancia", prompt, response, criteria)
        
        try:
            if "flan-t5" in self.model_name:
                llm_response = self.pipeline(eval_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            else:
                llm_response = self.pipeline(eval_prompt, max_length=len(eval_prompt) + 100, num_return_sequences=1)[0]['generated_text']
                llm_response = llm_response[len(eval_prompt):].strip()
            
            score, justification = self._extract_score_from_response(llm_response)
            
            return {
                "score": score,
                "feedback": f"[Relevance] {justification}",
                "details": {
                    "raw_response": llm_response,
                    "model_used": self.model_config.name
                }
            }
        except Exception as e:
            logger.error(f"Error en evaluación de relevancia: {str(e)}")
            return {
                "score": 5.0,
                "feedback": "[Relevance] Error en evaluación automática",
                "details": {"error": str(e)}
            }
    
    def evaluate_efficiency(self, prompt: str, response: str) -> Dict:
        """Evalúa la eficiencia de la respuesta."""
        criteria = "Concisión, claridad, organización, facilidad de comprensión, ausencia de información redundante."
        eval_prompt = self._generate_evaluation_prompt("Ed. Eficiencia", prompt, response, criteria)
        
        try:
            if "flan-t5" in self.model_name:
                llm_response = self.pipeline(eval_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            else:
                llm_response = self.pipeline(eval_prompt, max_length=len(eval_prompt) + 100, num_return_sequences=1)[0]['generated_text']
                llm_response = llm_response[len(eval_prompt):].strip()
            
            score, justification = self._extract_score_from_response(llm_response)
            
            return {
                "score": score,
                "feedback": f"[Efficiency] {justification}",
                "details": {
                    "raw_response": llm_response,
                    "model_used": self.model_config.name
                }
            }
        except Exception as e:
            logger.error(f"Error en evaluación de eficiencia: {str(e)}")
            return {
                "score": 5.0,
                "feedback": "[Efficiency] Error en evaluación automática",
                "details": {"error": str(e)}
            }
    
    def evaluate_creativity(self, prompt: str, response: str) -> Dict:
        """Evalúa la creatividad de la respuesta."""
        criteria = "Originalidad, innovación en el enfoque, diversidad de ideas, uso creativo del lenguaje, soluciones ingeniosas."
        eval_prompt = self._generate_evaluation_prompt("Dra. Creatividad", prompt, response, criteria)
        
        try:
            if "flan-t5" in self.model_name:
                llm_response = self.pipeline(eval_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            else:
                llm_response = self.pipeline(eval_prompt, max_length=len(eval_prompt) + 100, num_return_sequences=1)[0]['generated_text']
                llm_response = llm_response[len(eval_prompt):].strip()
            
            score, justification = self._extract_score_from_response(llm_response)
            
            return {
                "score": score,
                "feedback": f"[Creativity] {justification}",
                "details": {
                    "raw_response": llm_response,
                    "model_used": self.model_config.name
                }
            }
        except Exception as e:
            logger.error(f"Error en evaluación de creatividad: {str(e)}")
            return {
                "score": 5.0,
                "feedback": "[Creativity] Error en evaluación automática",
                "details": {"error": str(e)}
            }
    
    async def evaluate_response_async(self, prompt: str, response: str, weights: Optional[Dict[str, float]] = None) -> Dict:
        """Evalúa una respuesta usando todos los jueces de forma asíncrona."""
        if weights is None:
            weights = {
                "precision": 0.35,
                "coherence": 0.30,
                "relevance": 0.20,
                "efficiency": 0.10,
                "creativity": 0.05
            }
        
        # Ejecutar evaluaciones
        precision_result = self.evaluate_precision(prompt, response)
        coherence_result = self.evaluate_coherence(prompt, response)
        relevance_result = self.evaluate_relevance(prompt, response)
        efficiency_result = self.evaluate_efficiency(prompt, response)
        creativity_result = self.evaluate_creativity(prompt, response)
        
        # Calcular puntuación final
        final_score = (
            precision_result["score"] * weights["precision"] +
            coherence_result["score"] * weights["coherence"] +
            relevance_result["score"] * weights["relevance"] +
            efficiency_result["score"] * weights["efficiency"] +
            creativity_result["score"] * weights["creativity"]
        )
        
        # Calcular consenso (desviación estándar de las puntuaciones)
        scores = [
            precision_result["score"],
            coherence_result["score"],
            relevance_result["score"],
            efficiency_result["score"],
            creativity_result["score"]
        ]
        
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        consensus = max(0, 100 - (std_dev * 10))  # Consenso como porcentaje
        
        return {
            "overall_score": round(final_score, 1),
            "consensus": round(consensus, 0),
            "model_used": self.model_config.name,
            "individual_scores": {
                "precision": precision_result,
                "coherence": coherence_result,
                "relevance": relevance_result,
                "efficiency": efficiency_result,
                "creativity": creativity_result
            },
            "weights": weights,
            "evaluation_stats": {
                "mean_score": round(mean_score, 2),
                "std_deviation": round(std_dev, 2),
                "score_range": {
                    "min": min(scores),
                    "max": max(scores)
                }
            }
        }
    
    def evaluate_response(self, prompt: str, response: str, weights: Optional[Dict[str, float]] = None) -> Dict:
        """Versión síncrona de la evaluación."""
        import asyncio
        return asyncio.run(self.evaluate_response_async(prompt, response, weights))
    
    @classmethod
    def get_available_models(cls) -> Dict[str, ModelConfig]:
        """Retorna la lista de modelos disponibles."""
        return cls.AVAILABLE_MODELS
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[ModelConfig]:
        """Retorna información sobre un modelo específico."""
        return cls.AVAILABLE_MODELS.get(model_name)
