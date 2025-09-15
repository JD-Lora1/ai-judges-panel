"""
Contextual Judges - Evaluación Inteligente sin Dependencias
===========================================================

Sistema de evaluación que usa análisis contextual avanzado sin requerir
transformers ni otras dependencias pesadas.
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

class ContextualJudgesPanel:
    """
    Panel de jueces que usa análisis contextual avanzado para evaluación.
    No requiere transformers ni dependencias pesadas.
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
        
        self._evaluation_cache = {}
        logger.info(f"🧠 Evaluador contextual inicializado con {self.model_config.name}")
        logger.info(f"💡 Sistema optimizado sin dependencias pesadas - ultra-rápido!")
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extrae palabras clave relevantes del texto."""
        # Limpiar y tokenizar
        text_lower = text.lower()
        words = re.findall(r'\b\w{' + str(min_length) + ',}\b', text_lower)
        
        # Filtrar stop words comunes en español e inglés
        stop_words = {
            # Español
            'que', 'para', 'con', 'una', 'por', 'como', 'del', 'las', 'los', 'muy',
            'esto', 'esta', 'este', 'ser', 'tiene', 'puede', 'más', 'todo', 'pero',
            'son', 'sus', 'está', 'solo', 'hace', 'desde', 'sobre', 'entre', 'donde',
            # Inglés  
            'and', 'the', 'for', 'with', 'you', 'that', 'this', 'are', 'can', 'have',
            'will', 'not', 'but', 'all', 'from', 'they', 'been', 'has', 'were', 'said'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Contar frecuencias y retornar las más relevantes
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Ordenar por frecuencia y tomar las más importantes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:20]]  # Top 20 palabras
    
    def _calculate_semantic_overlap(self, prompt_keywords: List[str], response_keywords: List[str]) -> float:
        """Calcula el overlap semántico entre prompt y respuesta."""
        if not prompt_keywords or not response_keywords:
            return 0.0
        
        # Overlap directo
        direct_overlap = len(set(prompt_keywords) & set(response_keywords))
        
        # Overlap parcial (palabras que contienen otras o son similares)
        partial_overlap = 0
        for p_word in prompt_keywords:
            for r_word in response_keywords:
                if len(p_word) > 3 and len(r_word) > 3:
                    # Contención exacta
                    if p_word in r_word or r_word in p_word:
                        partial_overlap += 0.7
                        break
                    # Similitud por prefijos/sufijos
                    elif (p_word[:4] == r_word[:4] and len(p_word) > 4) or \
                         (p_word[-4:] == r_word[-4:] and len(p_word) > 4):
                        partial_overlap += 0.3
                        break
        
        # Overlap conceptual (palabras relacionadas semánticamente)
        concept_groups = [
            ['programa', 'programar', 'programación', 'código', 'software', 'desarrollo'],
            ['columna', 'tabla', 'base', 'datos', 'database', 'filtro', 'buscar'],
            ['web', 'página', 'sitio', 'internet', 'online', 'digital'],
            ['inteligencia', 'artificial', 'machine', 'learning', 'algoritmo', 'modelo'],
            ['design', 'thinking', 'metodología', 'agile', 'proceso', 'equipo']
        ]
        
        conceptual_overlap = 0
        for group in concept_groups:
            p_in_group = any(word in group for word in prompt_keywords)
            r_in_group = any(word in group for word in response_keywords)
            if p_in_group and r_in_group:
                conceptual_overlap += 0.5
        
        total_overlap = direct_overlap + partial_overlap + conceptual_overlap
        max_possible = max(len(prompt_keywords), len(response_keywords))
        
        return min(total_overlap / max_possible, 1.0)
    
    def _analyze_response_structure(self, response: str) -> Dict:
        """Analiza la estructura y formato de la respuesta."""
        structure_analysis = {}
        
        # Análisis de puntuación
        sentences = len(re.findall(r'[.!?]+', response))
        structure_analysis['sentence_count'] = sentences
        structure_analysis['has_structure'] = sentences > 0
        
        # Análisis de formato
        structure_analysis['has_bullet_points'] = bool(re.search(r'[•\-*]\s', response))
        structure_analysis['has_numbers'] = bool(re.search(r'\d+', response))
        structure_analysis['has_formatting'] = bool(re.search(r'\*\*.*\*\*|__.*__|`.*`', response))
        
        # Análisis de párrafos
        paragraphs = len([p for p in response.split('\n') if p.strip()])
        structure_analysis['paragraph_count'] = paragraphs
        structure_analysis['well_structured'] = paragraphs > 1 or sentences > 2
        
        return structure_analysis
    
    def _analyze_question_addressing(self, prompt: str, response: str) -> Dict:
        """Analiza si la respuesta realmente aborda la pregunta/solicitud."""
        question_analysis = {}
        
        # Detectar tipos de preguntas
        question_indicators = {
            'qué': ['qué', 'what'],
            'cómo': ['cómo', 'how'],
            'cuál': ['cuál', 'cuáles', 'which'],
            'cuándo': ['cuándo', 'when'],
            'dónde': ['dónde', 'where'],
            'por qué': ['por qué', 'porque', 'why'],
            'quién': ['quién', 'who']
        }
        
        prompt_lower = prompt.lower()
        question_type = None
        
        for q_type, indicators in question_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                question_type = q_type
                break
        
        question_analysis['question_type'] = question_type
        question_analysis['is_question'] = question_type is not None
        
        # Verificar si la respuesta es otra pregunta
        response_questions = response.count('?')
        response_statements = response.count('.')
        question_analysis['response_is_question'] = response_questions > response_statements
        
        # Verificar si aborda la pregunta
        if question_analysis['is_question'] and not question_analysis['response_is_question']:
            question_analysis['addresses_question'] = True
        else:
            question_analysis['addresses_question'] = not question_analysis['is_question']
        
        return question_analysis
    
    def _analyze_response_quality(self, prompt: str, response: str) -> Dict:
        """Análisis completo de calidad de respuesta."""
        
        # Análisis básico
        prompt_length = len(prompt.split())
        response_length = len(response.split())
        
        # Extraer palabras clave
        prompt_keywords = self._extract_keywords(prompt)
        response_keywords = self._extract_keywords(response)
        
        # Calcular overlap semántico
        semantic_overlap = self._calculate_semantic_overlap(prompt_keywords, response_keywords)
        
        # Análisis de estructura
        structure_analysis = self._analyze_response_structure(response)
        
        # Análisis de preguntas
        question_analysis = self._analyze_question_addressing(prompt, response)
        
        # Análisis de longitud
        length_ratio = response_length / max(prompt_length, 1)
        appropriate_length = 0.3 <= length_ratio <= 15  # Rango más amplio
        
        # Análisis de diversidad léxica
        unique_words = len(set(response_keywords))
        total_words = len(response_keywords)
        lexical_diversity = unique_words / max(total_words, 1)
        
        return {
            'semantic_overlap': semantic_overlap,
            'prompt_keywords': prompt_keywords,
            'response_keywords': response_keywords,
            'length_ratio': length_ratio,
            'appropriate_length': appropriate_length,
            'lexical_diversity': lexical_diversity,
            **structure_analysis,
            **question_analysis
        }
    
    def _smart_evaluate_aspect(self, prompt: str, response: str, aspect: str, analysis: Dict) -> Dict:
        """Evaluación inteligente por aspecto usando análisis contextual avanzado."""
        
        if aspect == "precision":
            # Precisión: ¿La respuesta es factualmente correcta y específica?
            base_score = 6.0
            
            if analysis['semantic_overlap'] < 0.1:
                base_score = 2.5
                feedback = f"Muy baja precisión - respuesta no relacionada con el prompt (overlap: {analysis['semantic_overlap']:.1%})"
            elif analysis['semantic_overlap'] < 0.2:
                base_score = 3.5
                feedback = f"Baja precisión - respuesta poco relacionada con el prompt (overlap: {analysis['semantic_overlap']:.1%})"
            elif analysis['addresses_question'] and analysis['has_structure']:
                base_score = 8.5
                feedback = "Alta precisión - respuesta específica y bien estructurada"
            elif analysis['addresses_question']:
                base_score = 7.5
                feedback = "Buena precisión - respuesta directa al prompt"
            elif analysis['semantic_overlap'] > 0.5:
                base_score = 7.0
                feedback = "Precisión adecuada - alto overlap semántico"
            else:
                base_score = 5.5
                feedback = "Precisión moderada - respuesta parcialmente relacionada"
        
        elif aspect == "coherence":
            # Coherencia: ¿La respuesta tiene lógica interna?
            base_score = 6.5
            
            coherence_score = 0
            if analysis['has_structure']: coherence_score += 2
            if analysis['well_structured']: coherence_score += 1.5
            if analysis['appropriate_length']: coherence_score += 1
            if analysis['sentence_count'] > 1: coherence_score += 1
            if not analysis['response_is_question']: coherence_score += 0.5
            
            base_score = min(4.0 + coherence_score, 9.0)
            
            if base_score >= 8:
                feedback = "Excelente coherencia - respuesta muy bien estructurada"
            elif base_score >= 7:
                feedback = "Buena coherencia - estructura lógica clara"
            elif base_score >= 6:
                feedback = "Coherencia adecuada - estructura básica presente"
            else:
                feedback = "Coherencia mejorable - estructura poco clara"
        
        elif aspect == "relevance":
            # Relevancia: ¿La respuesta es pertinente al prompt?
            overlap = analysis['semantic_overlap']
            
            if overlap >= 0.6:
                base_score = 9.0
                feedback = f"Muy relevante - excelente overlap semántico ({overlap:.1%})"
            elif overlap >= 0.4:
                base_score = 8.0
                feedback = f"Relevante - buen overlap semántico ({overlap:.1%})"
            elif overlap >= 0.25:
                base_score = 6.5
                feedback = f"Moderadamente relevante - overlap semántico aceptable ({overlap:.1%})"
            elif overlap >= 0.1:
                base_score = 4.0
                feedback = f"Parcialmente relevante - overlap semántico limitado ({overlap:.1%})"
            else:
                base_score = 2.0
                feedback = f"Baja relevancia - sin relación semántica clara ({overlap:.1%})"
                
            # Ajustar por tipo de pregunta
            if analysis['is_question'] and analysis['addresses_question']:
                base_score = min(base_score + 1.0, 9.5)
                feedback += " - aborda directamente la pregunta"
        
        elif aspect == "efficiency":
            # Eficiencia: ¿La respuesta es concisa y clara?
            base_score = 6.0
            
            if analysis['appropriate_length'] and analysis['addresses_question']:
                base_score = 8.0
                feedback = "Muy eficiente - longitud apropiada y directa"
            elif analysis['appropriate_length']:
                base_score = 7.0
                feedback = "Eficiente - longitud apropiada"
            elif analysis['length_ratio'] > 15:
                base_score = 3.5
                feedback = "Baja eficiencia - respuesta demasiado extensa"
            elif analysis['length_ratio'] < 0.2:
                base_score = 4.5
                feedback = "Eficiencia limitada - respuesta muy breve"
            else:
                base_score = 5.5
                feedback = "Eficiencia moderada"
                
            # Bonus por estructura clara
            if analysis['well_structured']:
                base_score = min(base_score + 0.5, 9.0)
        
        else:  # creativity
            # Creatividad: ¿La respuesta muestra originalidad?
            base_score = 5.5
            
            creativity_score = 0
            if analysis['lexical_diversity'] > 0.7: creativity_score += 2
            elif analysis['lexical_diversity'] > 0.5: creativity_score += 1
            
            if analysis['has_formatting']: creativity_score += 1
            if analysis['has_bullet_points']: creativity_score += 1
            if analysis['paragraph_count'] > 1: creativity_score += 0.5
            
            base_score = min(4.0 + creativity_score, 8.5)
            
            if base_score >= 7.5:
                feedback = "Alta creatividad - formato variado y vocabulario diverso"
            elif base_score >= 6.5:
                feedback = "Creatividad moderada - algunos elementos destacables"
            else:
                feedback = "Creatividad básica - respuesta estándar"
        
        return {
            "score": round(base_score, 1),
            "feedback": feedback,
            "details": {
                "analysis_method": "contextual_advanced",
                "semantic_overlap": analysis.get('semantic_overlap', 0),
                "key_factors": self._get_key_factors(aspect, analysis)
            }
        }
    
    def _get_key_factors(self, aspect: str, analysis: Dict) -> List[str]:
        """Obtiene los factores clave que influyeron en la evaluación."""
        factors = []
        
        if aspect == "relevance":
            overlap = analysis['semantic_overlap']
            if overlap > 0.5: factors.append("alto_overlap_semántico")
            elif overlap > 0.2: factors.append("overlap_semántico_moderado")
            else: factors.append("bajo_overlap_semántico")
            
            if analysis.get('addresses_question'): factors.append("aborda_pregunta")
            
        elif aspect == "coherence":
            if analysis.get('well_structured'): factors.append("bien_estructurado")
            if analysis.get('appropriate_length'): factors.append("longitud_apropiada")
            
        return factors
    
    def _generate_cache_key(self, prompt: str, response: str) -> str:
        """Genera una clave única para el cache."""
        content = f"{prompt}||{response}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    async def evaluate_response_async(self, prompt: str, response: str, weights: Optional[Dict[str, float]] = None) -> Dict:
        """Evalúa una respuesta usando análisis contextual avanzado."""
        
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
            # Análisis contextual avanzado
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
            
            # Calcular consenso real
            mean_score = sum(scores_list) / len(scores_list)
            variance = sum((score - mean_score) ** 2 for score in scores_list) / len(scores_list)
            std_dev = variance ** 0.5
            consensus = max(0, 100 - (std_dev * 12))  # Más sensible a variaciones
            
            evaluation_time = time.time() - start_time
            
            result = {
                "overall_score": round(overall_score, 1),
                "consensus": round(consensus, 0),
                "model_used": f"{self.model_config.name} (Contextual AI)",
                "individual_scores": individual_scores,
                "weights": weights,
                "evaluation_stats": {
                    "mean_score": round(mean_score, 2),
                    "std_deviation": round(std_dev, 2),
                    "score_range": {"min": min(scores_list), "max": max(scores_list)},
                    "semantic_overlap": round(analysis['semantic_overlap'], 3),
                    "evaluation_method": "contextual_advanced",
                    "evaluation_time": round(evaluation_time, 3)
                }
            }
            
            # Guardar en cache
            self._evaluation_cache[cache_key] = result
            
            logger.info(f"✅ Evaluación contextual completada en {evaluation_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en evaluación contextual: {str(e)}")
            # Fallback básico
            return {
                "overall_score": 6.0,
                "consensus": 70,
                "model_used": f"{self.model_config.name} (Fallback)",
                "individual_scores": {
                    aspect: {
                        "score": 6.0,
                        "feedback": f"[{aspect.capitalize()}] Error en evaluación - usando fallback",
                        "details": {"error": str(e)}
                    } for aspect in ["precision", "coherence", "relevance", "efficiency", "creativity"]
                },
                "weights": weights,
                "evaluation_stats": {
                    "mean_score": 6.0,
                    "std_deviation": 0.0,
                    "score_range": {"min": 6.0, "max": 6.0},
                    "evaluation_method": "fallback"
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
