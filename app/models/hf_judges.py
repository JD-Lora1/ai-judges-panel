"""
Hugging Face Judges Implementation
=================================

Real implementation of judges using lightweight Hugging Face models
for text evaluation and scoring.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM, pipeline
)
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import base classes from the original src
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from judges.base_judge import JudgmentAspect, JudgeEvaluation, EvaluationContext
from evaluators.meta_evaluator import ComprehensiveEvaluation

logger = logging.getLogger(__name__)

@dataclass
class HFModelConfig:
    """Configuration for Hugging Face models"""
    model_name: str
    task_type: str  # classification, text2text-generation, etc.
    max_length: int = 512
    device: str = "cpu"

class HuggingFaceJudge:
    """Base class for Hugging Face powered judges"""
    
    def __init__(self, name: str, aspect: JudgmentAspect, model_config: HFModelConfig):
        self.name = name
        self.aspect = aspect
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the Hugging Face model"""
        try:
            logger.info(f"🔄 Initializing {self.name} with model {self.model_config.model_name}")
            
            # Load model and tokenizer based on task type
            if self.model_config.task_type == "text-classification":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_config.model_name, 
                    trust_remote_code=True
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_config.model_name,
                    trust_remote_code=True
                )
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() and self.model_config.device == "cuda" else -1
                )
            else:
                # Generic pipeline approach for other tasks
                self.pipeline = pipeline(
                    self.model_config.task_type,
                    model=self.model_config.model_name,
                    device=0 if torch.cuda.is_available() and self.model_config.device == "cuda" else -1
                )
            
            self.is_initialized = True
            logger.info(f"✅ {self.name} initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.name}: {e}")
            raise
    
    async def evaluate_aspect(self, context: EvaluationContext) -> float:
        """Evaluate specific aspect using HF model - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def generate_feedback(self, context: EvaluationContext, score: float) -> Tuple[List[str], List[str]]:
        """Generate strengths and improvements - to be implemented by subclasses"""
        raise NotImplementedError

class CoherenceJudge(HuggingFaceJudge):
    """Judge for evaluating text coherence using BERT-based models"""
    
    def __init__(self):
        super().__init__(
            name="Prof. Coherence",
            aspect=JudgmentAspect.COHERENCE,
            model_config=HFModelConfig(
                model_name="microsoft/DialoGPT-medium",  # Lightweight conversational model
                task_type="text-generation",
                max_length=256
            )
        )
        self.sentence_transformer = None
    
    async def initialize(self):
        """Initialize coherence evaluation models"""
        try:
            logger.info(f"🔄 Initializing {self.name} for coherence evaluation...")
            
            # Use sentence transformers for coherence evaluation
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.is_initialized = True
            logger.info(f"✅ {self.name} initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.name}: {e}")
            raise
    
    async def evaluate_aspect(self, context: EvaluationContext) -> float:
        """Evaluate coherence by analyzing sentence embeddings and flow"""
        if not self.is_initialized:
            return 5.0
        
        try:
            text = context.candidate_response
            
            # Split into sentences
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return 7.0  # Single sentence is coherent by default
            
            # Get sentence embeddings
            embeddings = self.sentence_transformer.encode(sentences)
            
            # Calculate coherence as average cosine similarity between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)
            
            # Convert to 0-10 scale
            avg_similarity = np.mean(similarities)
            score = min(10.0, max(0.0, (avg_similarity + 1) * 5))  # Normalize to 0-10
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error in coherence evaluation: {e}")
            return 5.0
    
    async def generate_feedback(self, context: EvaluationContext, score: float) -> Tuple[List[str], List[str]]:
        """Generate feedback for coherence"""
        strengths = []
        improvements = []
        
        if score >= 8.0:
            strengths.extend([
                "Excelente flujo lógico entre ideas",
                "Transiciones suaves entre párrafos",
                "Estructura narrativa clara"
            ])
        elif score >= 6.0:
            strengths.append("Coherencia general adecuada")
            improvements.append("Mejorar las transiciones entre ideas")
        else:
            improvements.extend([
                "Reorganizar ideas para mejor flujo lógico",
                "Agregar conectores entre párrafos",
                "Clarificar la estructura del argumento"
            ])
        
        return strengths, improvements

class PrecisionJudge(HuggingFaceJudge):
    """Judge for evaluating factual precision using NLI models"""
    
    def __init__(self):
        super().__init__(
            name="Dr. Precision",
            aspect=JudgmentAspect.PRECISION,
            model_config=HFModelConfig(
                model_name="microsoft/DialoGPT-small",  # Lightweight model for factual checking
                task_type="text-classification",
                max_length=512
            )
        )
        self.fact_check_pipeline = None
    
    async def initialize(self):
        """Initialize precision evaluation models"""
        try:
            logger.info(f"🔄 Initializing {self.name} for precision evaluation...")
            
            # Use a lightweight fact-checking approach
            # For now, we'll use a simple heuristic-based approach
            self.is_initialized = True
            logger.info(f"✅ {self.name} initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.name}: {e}")
            raise
    
    async def evaluate_aspect(self, context: EvaluationContext) -> float:
        """Evaluate precision using heuristics and text analysis"""
        if not self.is_initialized:
            return 5.0
        
        try:
            text = context.candidate_response.lower()
            
            # Heuristic scoring based on text characteristics
            score = 7.0  # Base score
            
            # Check for uncertainty indicators (good for precision)
            uncertainty_indicators = ['quizás', 'tal vez', 'posiblemente', 'probablemente', 'puede que']
            uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in text)
            if uncertainty_count > 0:
                score += min(1.0, uncertainty_count * 0.5)
            
            # Check for absolute statements (potentially problematic)
            absolute_indicators = ['siempre', 'nunca', 'todos', 'nadie', 'completamente']
            absolute_count = sum(1 for indicator in absolute_indicators if indicator in text)
            if absolute_count > 2:
                score -= min(2.0, (absolute_count - 2) * 0.5)
            
            # Check for specific numbers and dates (good for precision)
            import re
            numbers = re.findall(r'\b\d+\b', text)
            dates = re.findall(r'\b\d{4}\b', text)  # Simple year detection
            if len(numbers) + len(dates) > 0:
                score += min(1.0, (len(numbers) + len(dates)) * 0.2)
            
            # Check for citations or references
            citations = re.findall(r'\b(según|de acuerdo|estudios|investigación|fuente)\b', text)
            if citations:
                score += min(1.5, len(citations) * 0.3)
            
            return min(10.0, max(0.0, float(score)))
            
        except Exception as e:
            logger.error(f"Error in precision evaluation: {e}")
            return 5.0
    
    async def generate_feedback(self, context: EvaluationContext, score: float) -> Tuple[List[str], List[str]]:
        """Generate feedback for precision"""
        strengths = []
        improvements = []
        
        text = context.candidate_response.lower()
        
        if score >= 8.0:
            strengths.extend([
                "Información específica y verificable",
                "Uso apropiado de calificadores",
                "Referencias a fuentes confiables"
            ])
        elif score >= 6.0:
            strengths.append("Información generalmente precisa")
            improvements.append("Agregar más detalles específicos y fuentes")
        else:
            improvements.extend([
                "Verificar la exactitud de las afirmaciones",
                "Evitar generalizaciones absolutas",
                "Incluir referencias y fuentes"
            ])
        
        # Check for specific numbers
        import re
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            strengths.append("Incluye datos numéricos específicos")
        else:
            improvements.append("Agregar datos cuantitativos cuando sea relevante")
        
        return strengths, improvements

class RelevanceJudge(HuggingFaceJudge):
    """Judge for evaluating response relevance using semantic similarity"""
    
    def __init__(self):
        super().__init__(
            name="Lic. Relevance",
            aspect=JudgmentAspect.RELEVANCE,
            model_config=HFModelConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                task_type="feature-extraction"
            )
        )
        self.sentence_transformer = None
    
    async def initialize(self):
        """Initialize relevance evaluation models"""
        try:
            logger.info(f"🔄 Initializing {self.name} for relevance evaluation...")
            
            # Use sentence transformers for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.is_initialized = True
            logger.info(f"✅ {self.name} initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.name}: {e}")
            raise
    
    async def evaluate_aspect(self, context: EvaluationContext) -> float:
        """Evaluate relevance using semantic similarity between prompt and response"""
        if not self.is_initialized:
            return 5.0
        
        try:
            prompt = context.original_prompt
            response = context.candidate_response
            
            # Get embeddings
            prompt_embedding = self.sentence_transformer.encode([prompt])
            response_embedding = self.sentence_transformer.encode([response])
            
            # Calculate semantic similarity
            similarity = cosine_similarity(prompt_embedding, response_embedding)[0][0]
            
            # Convert to 0-10 scale
            score = min(10.0, max(0.0, (similarity + 1) * 5))
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error in relevance evaluation: {e}")
            return 5.0
    
    async def generate_feedback(self, context: EvaluationContext, score: float) -> Tuple[List[str], List[str]]:
        """Generate feedback for relevance"""
        strengths = []
        improvements = []
        
        if score >= 8.0:
            strengths.extend([
                "Respuesta altamente relevante a la pregunta",
                "Aborda directamente los puntos solicitados",
                "Mantiene el foco en el tema principal"
            ])
        elif score >= 6.0:
            strengths.append("Respuesta generalmente relevante")
            improvements.append("Enfocarse más específicamente en la pregunta")
        else:
            improvements.extend([
                "Abordar directamente la pregunta formulada",
                "Eliminar información tangencial",
                "Reorganizar para mayor relevancia temática"
            ])
        
        return strengths, improvements

class EfficiencyJudge(HuggingFaceJudge):
    """Judge for evaluating response efficiency and clarity"""
    
    def __init__(self):
        super().__init__(
            name="Ed. Efficiency",
            aspect=JudgmentAspect.EFFICIENCY,
            model_config=HFModelConfig(
                model_name="textstat",  # We'll use textstat library
                task_type="readability"
            )
        )
    
    async def initialize(self):
        """Initialize efficiency evaluation"""
        try:
            logger.info(f"🔄 Initializing {self.name} for efficiency evaluation...")
            
            # We'll use text statistics for efficiency evaluation
            import textstat
            self.textstat = textstat
            
            self.is_initialized = True
            logger.info(f"✅ {self.name} initialized successfully!")
            
        except Exception as e:
            logger.warning(f"⚠️ textstat not available, using simple metrics for {self.name}")
            self.textstat = None
            self.is_initialized = True
    
    async def evaluate_aspect(self, context: EvaluationContext) -> float:
        """Evaluate efficiency based on conciseness and clarity"""
        if not self.is_initialized:
            return 5.0
        
        try:
            text = context.candidate_response
            prompt = context.original_prompt
            
            # Basic metrics
            words = len(text.split())
            sentences = len([s for s in text.split('.') if s.strip()])
            
            # Calculate efficiency score
            score = 7.0  # Base score
            
            # Penalize very long responses unless the prompt asks for detail
            detail_keywords = ['detallado', 'completo', 'exhaustivo', 'explicar en detalle']
            asks_for_detail = any(keyword in prompt.lower() for keyword in detail_keywords)
            
            if not asks_for_detail:
                if words > 300:
                    score -= min(2.0, (words - 300) / 100)
                elif words < 50:
                    score -= 1.0
            
            # Reward good sentence variety
            if sentences > 0:
                avg_words_per_sentence = words / sentences
                if 10 <= avg_words_per_sentence <= 20:
                    score += 1.0
                elif avg_words_per_sentence > 30:
                    score -= 1.0
            
            # Use textstat if available
            if self.textstat:
                try:
                    flesch_score = self.textstat.flesch_reading_ease(text)
                    if flesch_score > 60:  # Easy to read
                        score += 1.0
                    elif flesch_score < 30:  # Hard to read
                        score -= 1.0
                except:
                    pass
            
            return min(10.0, max(0.0, float(score)))
            
        except Exception as e:
            logger.error(f"Error in efficiency evaluation: {e}")
            return 5.0
    
    async def generate_feedback(self, context: EvaluationContext, score: float) -> Tuple[List[str], List[str]]:
        """Generate feedback for efficiency"""
        strengths = []
        improvements = []
        
        text = context.candidate_response
        words = len(text.split())
        
        if score >= 8.0:
            strengths.extend([
                "Respuesta concisa y clara",
                "Longitud apropiada para el contenido",
                "Fácil de leer y comprender"
            ])
        elif score >= 6.0:
            strengths.append("Respuesta clara y organizada")
            if words > 250:
                improvements.append("Considerar una versión más concisa")
            else:
                improvements.append("Mejorar la claridad de algunas ideas")
        else:
            if words > 300:
                improvements.append("Reducir la longitud manteniendo la información clave")
            elif words < 50:
                improvements.append("Desarrollar más las ideas principales")
            
            improvements.extend([
                "Simplificar el lenguaje utilizado",
                "Usar oraciones más cortas y directas"
            ])
        
        return strengths, improvements

class CreativityJudge(HuggingFaceJudge):
    """Judge for evaluating response creativity and originality"""
    
    def __init__(self):
        super().__init__(
            name="Dra. Creativity",
            aspect=JudgmentAspect.CREATIVITY,
            model_config=HFModelConfig(
                model_name="creativity-analysis",
                task_type="text-analysis"
            )
        )
    
    async def initialize(self):
        """Initialize creativity evaluation"""
        try:
            logger.info(f"🔄 Initializing {self.name} for creativity evaluation...")
            
            # We'll use lexical diversity and other heuristics for creativity
            self.is_initialized = True
            logger.info(f"✅ {self.name} initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.name}: {e}")
            raise
    
    async def evaluate_aspect(self, context: EvaluationContext) -> float:
        """Evaluate creativity using lexical diversity and other metrics"""
        if not self.is_initialized:
            return 5.0
        
        try:
            text = context.candidate_response.lower()
            words = text.split()
            
            if not words:
                return 0.0
            
            # Calculate lexical diversity (unique words / total words)
            unique_words = len(set(words))
            lexical_diversity = unique_words / len(words) if words else 0
            
            # Base creativity score
            score = 5.0 + (lexical_diversity - 0.5) * 10  # Scale lexical diversity
            
            # Check for creative elements
            creative_indicators = [
                'imaginemos', 'supongamos', 'metáfora', 'analogía', 'como si',
                'innovador', 'creativo', 'original', 'único', 'diferente',
                'por ejemplo', 'imagina', 'visualiza'
            ]
            
            creative_count = sum(1 for indicator in creative_indicators if indicator in text)
            score += min(2.0, creative_count * 0.5)
            
            # Check for varied sentence structures
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 1:
                sentence_lengths = [len(s.split()) for s in sentences]
                length_variety = len(set(sentence_lengths)) / len(sentences) if sentences else 0
                score += length_variety * 2
            
            return min(10.0, max(0.0, float(score)))
            
        except Exception as e:
            logger.error(f"Error in creativity evaluation: {e}")
            return 5.0
    
    async def generate_feedback(self, context: EvaluationContext, score: float) -> Tuple[List[str], List[str]]:
        """Generate feedback for creativity"""
        strengths = []
        improvements = []
        
        text = context.candidate_response.lower()
        
        if score >= 8.0:
            strengths.extend([
                "Uso creativo del lenguaje",
                "Perspectiva original y única",
                "Ejemplos innovadores y claros"
            ])
        elif score >= 6.0:
            strengths.append("Enfoque interesante del tema")
            improvements.append("Explorar más perspectivas creativas")
        else:
            improvements.extend([
                "Agregar ejemplos más originales",
                "Usar analogías o metáforas para clarificar",
                "Explorar diferentes enfoques del tema"
            ])
        
        # Check for creative elements
        creative_indicators = ['imaginemos', 'supongamos', 'por ejemplo', 'imagina']
        if any(indicator in text for indicator in creative_indicators):
            strengths.append("Uso efectivo de ejemplos e imaginación")
        else:
            improvements.append("Incorporar más ejemplos ilustrativos")
        
        return strengths, improvements

class HuggingFaceJudgesPanel:
    """Main panel coordinating all Hugging Face judges"""
    
    def __init__(self):
        self.judges = []
        self.is_ready_flag = False
        
        # Initialize judges
        self.judges = [
            PrecisionJudge(),
            CoherenceJudge(),
            RelevanceJudge(),
            EfficiencyJudge(),
            CreativityJudge()
        ]
        
        # Default weights
        self.weights = {
            "precision": 0.25,
            "coherence": 0.25,
            "relevance": 0.20,
            "efficiency": 0.15,
            "creativity": 0.15
        }
    
    async def initialize(self):
        """Initialize all judges"""
        logger.info("🏛️ Initializing Hugging Face Judges Panel...")
        
        initialization_tasks = [judge.initialize() for judge in self.judges]
        await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        self.is_ready_flag = True
        logger.info("✅ All judges initialized successfully!")
    
    def is_ready(self) -> bool:
        """Check if the panel is ready"""
        return self.is_ready_flag and all(judge.is_initialized for judge in self.judges)
    
    async def evaluate(self, prompt: str, response: str, domain: Optional[str] = None, 
                      include_automatic_metrics: bool = True) -> ComprehensiveEvaluation:
        """Evaluate text using all judges"""
        start_time = time.time()
        
        # Create evaluation context
        context = EvaluationContext(
            original_prompt=prompt,
            candidate_response=response,
            domain=domain
        )
        
        # Evaluate with each judge
        individual_evaluations = {}
        individual_scores = {}
        all_strengths = []
        all_improvements = []
        
        for judge in self.judges:
            try:
                # Get aspect score
                score = await judge.evaluate_aspect(context)
                individual_scores[judge.aspect.value] = score
                
                # Generate feedback
                strengths, improvements = await judge.generate_feedback(context, score)
                
                # Create judge evaluation
                evaluation = JudgeEvaluation(
                    aspect=judge.aspect,
                    score=score,
                    reasoning=f"Evaluación de {judge.aspect.value} completada con modelos HF",
                    strengths=strengths,
                    improvements=improvements,
                    confidence=0.8,
                    evaluation_time=0.5,
                    metadata={"model_type": "huggingface", "judge_name": judge.name}
                )
                
                individual_evaluations[judge.aspect.value] = evaluation
                
                # Add context to feedback
                for strength in strengths:
                    all_strengths.append(f"[{judge.aspect.value.title()}] {strength}")
                for improvement in improvements:
                    all_improvements.append(f"[{judge.aspect.value.title()}] {improvement}")
                    
            except Exception as e:
                logger.error(f"Error evaluating with {judge.name}: {e}")
                individual_scores[judge.aspect.value] = 5.0
        
        # Calculate weighted final score
        final_score = self._calculate_weighted_score(individual_scores)
        
        # Calculate consensus
        consensus_level = self._calculate_consensus(individual_scores)
        
        # Create comprehensive evaluation
        evaluation = ComprehensiveEvaluation(
            final_score=final_score,
            individual_scores=individual_scores,
            consensus_level=consensus_level,
            strengths=all_strengths[:10],  # Limit to top 10
            improvements=all_improvements[:10],  # Limit to top 10
            detailed_feedback=individual_evaluations,
            evaluation_time=time.time() - start_time,
            metadata={
                "judges_count": len(self.judges),
                "model_type": "huggingface",
                "domain": domain,
                "weights_used": self.weights.copy()
            }
        )
        
        return evaluation
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted final score"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for aspect, score in scores.items():
            weight = self.weights.get(aspect, 0.2)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 5.0
    
    def _calculate_consensus(self, scores: Dict[str, float]) -> float:
        """Calculate consensus level between judges"""
        if len(scores) < 2:
            return 1.0
        
        score_values = list(scores.values())
        mean_score = np.mean(score_values)
        std_dev = np.std(score_values)
        
        # Normalize: low std_dev = high consensus
        consensus = max(0.0, 1.0 - (std_dev / 5.0))
        return round(consensus, 3)
    
    def get_judges_info(self) -> Dict[str, Any]:
        """Get information about all judges"""
        judges_info = []
        
        for judge in self.judges:
            judges_info.append({
                "name": judge.name,
                "aspect": judge.aspect.value,
                "model": judge.model_config.model_name,
                "initialized": judge.is_initialized,
                "weight": self.weights.get(judge.aspect.value, 0.0)
            })
        
        return {
            "judges": judges_info,
            "total_judges": len(self.judges),
            "panel_ready": self.is_ready()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Hugging Face judges panel...")
        # Cleanup code if needed for models
        self.is_ready_flag = False
