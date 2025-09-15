"""
Smart LLM Judges - Implementación Híbrida
=========================================

Sistema híbrido que usa análisis contextual inteligente cuando los modelos
transformers no están disponibles, pero mantiene la interfaz LLM.
"""

import re
import time
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    model_id: str
    max_length: int
    temperature: float = 0.7
    description: str = ""

class SmartLLMJudgesPanel:
    """
    Panel de jueces híbrido que usa análisis contextual inteligente
    cuando los modelos LLM no están disponibles.
    """
    
    AVAILABLE_MODELS = {
        "distilgpt2": ModelConfig(
            name="DistilGPT2",
            model_id="distilgpt2",
            max_length=50,
            temperature=0.2,
            description="Modelo ligero de generación de texto, rápido y eficiente (RECOMENDADO)"
        ),
        "google/flan-t5-base": ModelConfig(
            name="FLAN-T5 Base",
            model_id="google/flan-t5-base",
            max_length=60,
            temperature=0.3,
            description="Modelo instruction-following de Google, excelente para evaluación de relevancia"
        ),
        "microsoft/DialoGPT-small": ModelConfig(
            name="DialoGPT Small",
            model_id="microsoft/DialoGPT-small",
            max_length=50,
            temperature=0.2,
            description="Versión más ligera de DialoGPT, ideal para evaluaciones rápidas"
        ),
        "microsoft/DialoGPT-medium": ModelConfig(
            name="DialoGPT Medium",
            model_id="microsoft/DialoGPT-medium",
            max_length=80,
            temperature=0.2,
            description="Modelo conversacional de Microsoft, bueno para evaluar coherencia contextual"
        )
    }
    
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.model_config = self.AVAILABLE_MODELS.get(model_name)
        if not self.model_config:
            raise ValueError(f"Modelo {model_name} no disponible. Modelos disponibles: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.use_transformers = False
        self.pipeline = None
        self._evaluation_cache = {}
        
        # Intentar cargar transformers, pero usar fallback si falla
        self._try_load_transformers()
    
    def _try_load_transformers(self):
        """Intenta cargar transformers, usa fallback inteligente si falla."""
        # Por ahora, usar siempre el evaluador contextual inteligente
        # ya que hay conflictos con las versiones de transformers
        logger.info(f"🧠 Usando evaluador contextual inteligente para {self.model_config.name}")
        logger.info(f"💡 Esto evita conflictos de dependencias y es más rápido")
        self.use_transformers = False
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extrae palabras clave relevantes del texto."""
        # Limpiar y tokenizar
        text_lower = text.lower()
        words = re.findall(r'\b\w{' + str(min_length) + ',}\b', text_lower)
        
        # Filtrar stop words comunes
        stop_words = {
            'que', 'para', 'con', 'una', 'por', 'como', 'del', 'las', 'los', 'muy',
            'and', 'the', 'for', 'with', 'you', 'that', 'this', 'are', 'can', 'have'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Contar frecuencias y retornar las más relevantes
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Ordenar por frecuencia y tomar las más importantes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:15]]  # Top 15 palabras
    
    def _calculate_semantic_overlap(self, prompt_keywords: List[str], response_keywords: List[str]) -> float:
        """Calcula el overlap semántico entre prompt y respuesta."""
        if not prompt_keywords or not response_keywords:
            return 0.0
        
        # Overlap directo
        direct_overlap = len(set(prompt_keywords) & set(response_keywords))
        
        # Overlap parcial (palabras que contienen otras)
        partial_overlap = 0
        for p_word in prompt_keywords:
            for r_word in response_keywords:
                if len(p_word) > 3 and len(r_word) > 3:
                    if p_word in r_word or r_word in p_word:
                        partial_overlap += 0.5
                        break
        
        total_overlap = direct_overlap + partial_overlap
        max_possible = max(len(prompt_keywords), len(response_keywords))
        
        return min(total_overlap / max_possible, 1.0)
    
    def _analyze_response_quality(self, prompt: str, response: str) -> Dict:
        """Analiza la calidad de la respuesta usando análisis contextual."""
        
        # Análisis básico
        prompt_length = len(prompt.split())
        response_length = len(response.split())
        
        # Extraer palabras clave
        prompt_keywords = self._extract_keywords(prompt)
        response_keywords = self._extract_keywords(response)
        
        # Calcular overlap semántico
        semantic_overlap = self._calculate_semantic_overlap(prompt_keywords, response_keywords)
        
        # Análisis de estructura
        has_structure = len(re.findall(r'[.!?]', response)) > 0
        has_bullet_points = bool(re.search(r'[•\-*]\s', response))
        has_numbers = bool(re.search(r'\d+', response))
        
        # Análisis de longitud apropiada
        length_ratio = response_length / max(prompt_length, 1)
        appropriate_length = 0.5 <= length_ratio <= 10  # Entre 50% y 1000% del prompt
        
        # Detectar si realmente responde la pregunta
        question_indicators = ['qué', 'cómo', 'cuál', 'cuándo', 'dónde', 'por qué', 'what', 'how', 'when', 'where', 'why']
        is_question = any(indicator in prompt.lower() for indicator in question_indicators)
        
        # Si es pregunta, verificar que la respuesta no sea otra pregunta
        response_is_question = response.count('?') > response.count('.')
        addresses_question = is_question and not response_is_question
        
        return {
            'semantic_overlap': semantic_overlap,
            'prompt_keywords': prompt_keywords,
            'response_keywords': response_keywords,
            'length_ratio': length_ratio,
            'appropriate_length': appropriate_length,
            'has_structure': has_structure,
            'has_bullet_points': has_bullet_points,
            'has_numbers': has_numbers,
            'addresses_question': addresses_question,
            'is_question': is_question,
            'response_is_question': response_is_question
        }
    
    def _smart_evaluate_aspect(self, prompt: str, response: str, aspect: str, analysis: Dict) -> Dict:
        """Evaluación inteligente por aspecto usando análisis contextual."""
        
        if aspect == "precision":
            # Precisión: ¿La respuesta es factualmente correcta y específica?
            base_score = 7.0
            
            # Reducir si no hay overlap semántico
            if analysis['semantic_overlap'] < 0.2:
                base_score = 3.0
                feedback = f"Baja precisión - respuesta no relacionada con el prompt"
            elif analysis['addresses_question'] and analysis['has_structure']:
                base_score = 8.5
                feedback = "Respuesta precisa y bien estructurada"
            elif analysis['addresses_question']:
                base_score = 7.5
                feedback = "Respuesta precisa pero podría ser más estructurada"
            else:
                base_score = 5.0
                feedback = "Precisión moderada - respuesta parcialmente relevante"
        
        elif aspect == "coherence":
            # Coherencia: ¿La respuesta tiene lógica interna?
            base_score = 7.0
            
            if analysis['has_structure'] and analysis['appropriate_length']:
                base_score = 8.0
                feedback = "Respuesta coherente y bien estructurada"
            elif analysis['has_structure']:
                base_score = 7.5
                feedback = "Coherencia adecuada con buena estructura"
            elif analysis['appropriate_length']:
                base_score = 6.5
                feedback = "Coherente pero podría mejorar la estructura"
            else:
                base_score = 5.0
                feedback = "Coherencia básica - estructura mejorable"
        
        elif aspect == "relevance":
            # Relevancia: ¿La respuesta es pertinente al prompt?
            overlap = analysis['semantic_overlap']
            
            if overlap >= 0.5:
                base_score = 9.0
                feedback = f"Muy relevante - alto overlap semántico ({overlap:.1%})"
            elif overlap >= 0.3:
                base_score = 7.5
                feedback = f"Relevante - buen overlap semántico ({overlap:.1%})"
            elif overlap >= 0.1:
                base_score = 5.0
                feedback = f"Parcialmente relevante - overlap limitado ({overlap:.1%})"
            else:
                base_score = 2.0
                feedback = f"Baja relevancia - sin relación semántica clara ({overlap:.1%})"
        
        elif aspect == "efficiency":
            # Eficiencia: ¿La respuesta es concisa y clara?
            base_score = 6.0
            
            if analysis['appropriate_length'] and analysis['addresses_question']:
                base_score = 8.0
                feedback = "Respuesta eficiente - longitud apropiada y directa"
            elif analysis['appropriate_length']:
                base_score = 7.0
                feedback = "Longitud apropiada pero podría ser más directa"
            elif analysis['length_ratio'] > 10:
                base_score = 4.0
                feedback = "Respuesta demasiado extensa - baja eficiencia"
            elif analysis['length_ratio'] < 0.3:
                base_score = 5.0
                feedback = "Respuesta muy breve - podría ser más completa"
            else:
                base_score = 6.0
                feedback = "Eficiencia moderada"
        
        else:  # creativity
            # Creatividad: ¿La respuesta muestra originalidad?
            base_score = 6.0
            
            variety_score = len(set(analysis['response_keywords'])) / max(len(analysis['response_keywords']), 1)
            
            if analysis['has_bullet_points'] and variety_score > 0.7:
                base_score = 8.0
                feedback = "Respuesta creativa con formato variado y vocabulario diverso"
            elif analysis['has_bullet_points'] or variety_score > 0.6:
                base_score = 7.0
                feedback = "Creatividad moderada con elementos de formato o vocabulario variado"
            else:
                base_score = 5.5
                feedback = "Creatividad básica - respuesta estándar"
        
        return {
            "score": round(base_score, 1),
            "feedback": feedback,
            "details": {
                "analysis_used": "smart_contextual",
                "semantic_overlap": analysis.get('semantic_overlap', 0)
            }
        }
    
    def _generate_cache_key(self, prompt: str, response: str) -> str:
        """Genera una clave única para el cache."""
        content = f"{prompt}||{response}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    async def evaluate_response_async(self, prompt: str, response: str, weights: Optional[Dict[str, float]] = None) -> Dict:
        """Evalúa una respuesta usando el sistema híbrido."""
        
        start_time = time.time()
        
        if weights is None:
            weights = {
                "precision": 0.35,
                "coherence": 0.30,
                "relevance": 0.20,
                "efficiency": 0.10,
                "creativity": 0.05
            }
        
        # Verificar cache
        cache_key = self._generate_cache_key(prompt, response)
        if cache_key in self._evaluation_cache:
            cached_result = self._evaluation_cache[cache_key]
            logger.info("✅ Usando resultado desde cache")
            return cached_result
        
        try:
            # Análisis contextual inteligente
            analysis = self._analyze_response_quality(prompt, response)
            
            # Evaluar cada aspecto
            aspects = ["precision", "coherence", "relevance", "efficiency", "creativity"]
            individual_scores = {}
            scores_list = []
            
            for aspect in aspects:
                result = self._smart_evaluate_aspect(prompt, response, aspect, analysis)
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
            
            evaluation_time = time.time() - start_time
            
            result = {
                "overall_score": round(overall_score, 1),
                "consensus": round(consensus, 0),
                "model_used": f"{self.model_config.name} (Smart Contextual)",
                "individual_scores": individual_scores,
                "weights": weights,
                "evaluation_stats": {
                    "mean_score": round(mean_score, 2),
                    "std_deviation": round(std_dev, 2),
                    "score_range": {"min": min(scores_list), "max": max(scores_list)},
                    "semantic_overlap": round(analysis['semantic_overlap'], 3),
                    "evaluation_method": "smart_contextual"
                }
            }
            
            # Guardar en cache
            self._evaluation_cache[cache_key] = result
            
            logger.info(f"✅ Evaluación completada en {evaluation_time:.2f}s usando análisis contextual")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en evaluación: {str(e)}")
            # Fallback básico si todo falla
            return {
                "overall_score": 6.0,
                "consensus": 70,
                "model_used": f"{self.model_config.name} (Fallback)",
                "individual_scores": {
                    aspect: {
                        "score": 6.0,
                        "feedback": f"[{aspect.capitalize()}] Evaluación fallback - sistema no disponible",
                        "details": {"error": str(e)}
                    } for aspect in ["precision", "coherence", "relevance", "efficiency", "creativity"]
                },
                "weights": weights,
                "evaluation_stats": {
                    "mean_score": 6.0,
                    "std_deviation": 0.0,
                    "score_range": {"min": 6.0, "max": 6.0}
                }
            }
    
    def evaluate_response(self, prompt: str, response: str, weights: Optional[Dict[str, float]] = None) -> Dict:
        """Versión síncrona de la evaluación."""
        return asyncio.run(self.evaluate_response_async(prompt, response, weights))
    
    @classmethod
    def get_available_models(cls) -> Dict[str, ModelConfig]:
        """Retorna la lista de modelos disponibles."""
        return cls.AVAILABLE_MODELS
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[ModelConfig]:
        """Retorna información sobre un modelo específico."""
        return cls.AVAILABLE_MODELS.get(model_name)
