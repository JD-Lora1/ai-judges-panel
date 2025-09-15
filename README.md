# AI Judges Panel 🏛️⚖️
## Una Arquitectura Multi-Agente para Evaluación de LLMs

🚀 **[Aplicación Web en Vivo - https://ai-judges-ai.up.railway.app/](https://ai-judges-ai.up.railway.app/)**

### 🎯 Visión del Proyecto

Este proyecto implementa una **arquitectura de panel de jueces** donde múltiples LLMs especializados evalúan las respuestas de otros LLMs desde diferentes perspectivas, combinando la robustez de métricas automáticas con el juicio inteligente de modelos de lenguaje.

## 🏗️ Arquitectura: Panel de Jueces Especializados

```
┌─────────────────────────────────────────────────────────────────┐
│                      SISTEMA DE EVALUACIÓN                      │
└─────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            ┌───────▼─────────┐           ┌───────▼─────────┐
            │  LLM CANDIDATO  │           │ MÉTRICAS AUTO   │
            │   (A evaluar)   │           │ BLEU/ROUGE/BERT │
            └─────────────────┘           └─────────────────┘
                    │                             │
            ┌───────▼─────────────────────────────▼─────────┐
            │            PANEL DE JUECES                    │
            ├───────────────────────────────────────────────┤
            │  🎯 Juez de PRECISIÓN     📊 Score: 0-10     │
            │  🎨 Juez de CREATIVIDAD   📊 Score: 0-10     │
            │  🧠 Juez de COHERENCIA    📊 Score: 0-10     │
            │  🎪 Juez de RELEVANCIA    📊 Score: 0-10     │
            │  ⚡ Juez de EFICIENCIA    📊 Score: 0-10     │
            └───────────────┬───────────────────────────────┘
                            │
                    ┌───────▼─────────┐
                    │ META-EVALUADOR  │
                    │   (Agregador)   │
                    └─────────────────┘
                            │
                    ┌───────▼─────────┐
                    │ REPORTE FINAL   │
                    │ Score + Insights│
                    └─────────────────┘
```

## 🧠 Especialización de Jueces

### 🎯 **Juez de Precisión** (`PrecisionJudge`)
- **Rol**: Evalúa factualidad y exactitud de la información
- **Criterios**: Veracidad, ausencia de alucinaciones, citas correctas
- **Prompt especializado**: "Como experto en verificación de hechos..."

### 🎨 **Juez de Creatividad** (`CreativityJudge`)
- **Rol**: Evalúa originalidad, innovación y pensamiento lateral
- **Criterios**: Novedad de ideas, perspectivas únicas, soluciones creativas
- **Prompt especializado**: "Como crítico de arte y literatura..."

### 🧠 **Juez de Coherencia** (`CoherenceJudge`)
- **Rol**: Evalúa lógica interna, estructura y fluidez
- **Criterios**: Argumentación sólida, transiciones suaves, consistencia
- **Prompt especializado**: "Como filósofo especializado en lógica..."

### 🎪 **Juez de Relevancia** (`RelevanceJudge`)
- **Rol**: Evalúa si la respuesta aborda realmente la pregunta
- **Criterios**: Pertinencia, completitud de la respuesta
- **Prompt especializado**: "Como evaluador de comprensión lectora..."

### ⚡ **Juez de Eficiencia** (`EfficiencyJudge`)
- **Rol**: Evalúa concisión y claridad comunicativa
- **Criterios**: Brevedad sin pérdida de información, claridad
- **Prompt especializado**: "Como editor profesional..."

## 📊 Sistema de Agregación

### **Meta-Evaluador** (`MetaEvaluator`)
- Combina scores de todos los jueces
- Aplica pesos personalizables según el contexto
- Genera un reporte explicativo detallado
- Identifica consensos y discrepancias entre jueces

### **Métricas Híbridas Ponderadas**
```python
final_score = (
    0.35 * precision_score +    # Mayor peso: exactitud factual
    0.30 * coherence_score +    # Mayor peso: estructura lógica  
    0.20 * relevance_score +    # Peso medio: pertinencia al prompt
    0.10 * efficiency_score +   # Peso menor: claridad y concisión
    0.05 * creativity_score     # Peso mínimo: originalidad
) + automatic_metrics_boost
```

> **Nota**: El sistema evalúa la **relación prompt-respuesta**, no solo la respuesta aisladamente. Cada juez considera tanto la pregunta original como la calidad de la respuesta en ese contexto específico.

## 🚀 Casos de Uso

### 1. **Evaluación de Chatbots** 🤖
- **Prompt-Aware**: Compara cómo GPT, Claude, Gemini responden al **mismo prompt**
- **Contextual**: Identifica qué modelo entiende mejor la **intención** del prompt
- **Optimización**: Mejora prompts basado en feedback específico de cada relación prompt-respuesta

### 2. **Content Generation Assessment** ✍️
- **Pertinencia**: ¿El contenido generado cumple **exactamente** con el briefing/prompt?
- **Coherencia Contextual**: ¿La estructura responde a lo solicitado en el prompt?
- **Precisión**: ¿Los hechos/datos generados son consistentes con los requerimientos?

### 3. **Educational AI Evaluation** 🎓
- **Relevancia Pedagógica**: ¿La explicación del AI aborda la pregunta del estudiante?
- **Coherencia Didáctica**: ¿La respuesta sigue un flujo lógico apropiado para el nivel?
- **Precisión Educativa**: ¿La información es factualmente correcta y verificable?

### 4. **Research Assistant Analysis** 🔬
- **Relevancia de Síntesis**: ¿La síntesis responde a la consulta de investigación?
- **Precisión Académica**: Verifica accuracy de citaciones **en relación al tema** consultado
- **Coherencia Conceptual**: ¿Las conexiones propuestas son lógicas dado el contexto?

### 5. **Evaluación de Prompts Complejos** 🧠
- **Multi-step Instructions**: Evalúa cómo los LLMs manejan prompts con múltiples pasos
- **Domain-Specific Queries**: Mide precisión en respuestas técnicas, legales, médicas
- **Creative vs Factual Balance**: Detecta cuándo priorizar creatividad vs exactitud según el prompt

## 🛠️ Estructura del Proyecto

```
ai-judges-panel/
├── src/
│   ├── judges/
│   │   ├── base_judge.py          # Clase base para todos los jueces
│   │   ├── precision_judge.py     # Juez especializado en precisión
│   │   ├── creativity_judge.py    # Juez especializado en creatividad
│   │   ├── coherence_judge.py     # Juez especializado en coherencia
│   │   ├── relevance_judge.py     # Juez especializado en relevancia
│   │   └── efficiency_judge.py    # Juez especializado en eficiencia
│   ├── evaluators/
│   │   ├── meta_evaluator.py      # Agregador inteligente de scores
│   │   ├── hybrid_metrics.py      # Combina métricas auto + LLM
│   │   └── report_generator.py    # Generador de reportes detallados
│   ├── models/
│   │   ├── llm_interface.py       # Interfaz unificada para LLMs
│   │   └── candidate_model.py     # Wrapper para modelos a evaluar
│   └── utils/
│       ├── prompts.py             # Prompts especializados
│       ├── scoring.py             # Sistema de puntuación
│       └── config.py              # Configuraciones del sistema
├── notebooks/
│   ├── 01_demo_basic.ipynb        # Demo básico del sistema
│   ├── 02_judge_comparison.ipynb  # Comparación entre jueces
│   ├── 03_model_benchmarking.ipynb# Benchmark de modelos
│   └── 04_custom_evaluation.ipynb # Evaluaciones personalizadas
├── examples/
│   ├── chatbot_evaluation.py      # Ejemplo: evaluar chatbots
│   ├── creative_writing.py        # Ejemplo: evaluar escritura creativa
│   └── code_generation.py         # Ejemplo: evaluar código generado
├── tests/
│   ├── test_judges.py             # Tests para jueces individuales
│   ├── test_meta_evaluator.py     # Tests para meta-evaluador
│   └── test_integration.py        # Tests de integración completa
├── docs/
│   ├── architecture.md            # Documentación arquitectónica
│   ├── judge_design.md            # Diseño de jueces especializados
│   └── evaluation_guide.md        # Guía de uso y evaluación
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

## 🎯 Ventajas de Esta Arquitectura

### **1. Evaluación Multi-Dimensional** 📐
- Cada aspecto evaluado por un especialista
- Evita sesgos de evaluación monolítica
- Perspectivas complementarias

### **2. Interpretabilidad Completa** 🔍
- Cada juez explica su calificación
- Feedback específico y accionable
- Transparencia en el proceso de evaluación

### **3. Escalabilidad Modular** 📈
- Fácil agregar nuevos jueces especializados
- Pesos ajustables según contexto
- Adaptable a diferentes dominios

### **4. Robustez Contra Sesgos** 🛡️
- Múltiples perspectivas reducen sesgos individuales
- Consenso emergente más confiable
- Detección automática de discrepancias

## 🚀 Quick Start

### Opción 1: Aplicación Web (Recomendado)

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación web
uvicorn app.main:app --reload

# Abrir en navegador: http://localhost:8000
```

### Opción 2: API Programática

```python
import asyncio
from app.models.hf_judges import HuggingFaceJudgesPanel

# Inicializar el panel de jueces con modelos de HuggingFace
async def main():
    panel = HuggingFaceJudgesPanel()
    await panel.initialize()
    
    # Evaluar relación prompt-respuesta (no solo la respuesta)
    result = await panel.evaluate(
        prompt="Explica qué es la inteligencia artificial y da 3 ejemplos prácticos",
        response="La IA es un campo de la informática que simula inteligencia humana. Ejemplos: 1) Asistentes virtuales como Siri, 2) Recomendaciones de Netflix, 3) Coches autónomos de Tesla.",
        domain="technical"
    )
    
    print(f"Score Final: {result.final_score:.1f}/10")
    print(f"Consenso: {result.consensus_level:.1%}")
    print("\nEvaluación por aspecto:")
    for aspect, score in result.individual_scores.items():
        print(f"  {aspect.title()}: {score:.1f}/10")
    print("\nFortalezas principales:")
    for strength in result.strengths[:3]:
        print(f"  ✓ {strength}")

# Ejecutar
asyncio.run(main())
```

### Opción 3: Notebooks de Investigación

```bash
# Ejecutar notebooks originales
jupyter notebook notebooks/01_demo_basic.ipynb
```

## 🎪 ¿Por Qué Esta Arquitectura es Innovadora?

### **Diferenciadores Clave:**

1. **Especialización Inteligente**: Cada juez tiene expertise específico
2. **Híbrida**: Combina métricas automáticas + evaluación por LLM
3. **Explicable**: No solo scores, sino razones detalladas
4. **Consensual**: Múltiples perspectivas convergen en evaluación final
5. **Adaptativa**: Pesos ajustables según tipo de evaluación

---

### 🎊 **¡Listo para Evaluar el Futuro de la IA!**

*Este proyecto representa la evolución natural de la evaluación de IA: de métricas simples a juicios inteligentes especializados.*

## 🚀 Despliegue en Railway

### Despliegue Automático

1. **Fork el repositorio** en GitHub
2. **Conecta con Railway**: 
   - Ve a [railway.app](https://railway.app)
   - Conecta tu cuenta de GitHub
   - Selecciona este repositorio
3. **Configuración automática**:
   - Railway detectará automáticamente el `railway.toml`
   - La aplicación se desplegará con FastAPI + Uvicorn
   - Los modelos de HuggingFace se cargarán en el primer arranque

### Variables de Entorno (Opcionales)

```bash
ENVIRONMENT=production
PYTHONPATH=.
# HF_TOKEN=your_huggingface_token  # Si usas modelos privados
```

### Endpoints Disponibles

Desde la aplicación desplegada: **https://ai-judges-ai.up.railway.app/**

- `GET /` - [Interfaz web principal](https://ai-judges-ai.up.railway.app/)
- `GET /evaluate` - [Página de evaluación interactiva](https://ai-judges-ai.up.railway.app/evaluate)
- `POST /api/v1/evaluate` - API de evaluación básica
- `POST /api/v1/evaluate/detailed` - API de evaluación avanzada
- `POST /api/v1/evaluate/compare` - Comparación de respuestas
- `GET /api/v1/docs` - [Documentación interactiva de la API](https://ai-judges-ai.up.railway.app/api/v1/docs)
- `GET /health` - Health check para Railway

**Próximo paso**: Despliega en Railway o ejecuta `uvicorn app.main:app --reload`

## 🤖 **Cómo Funciona: Arquitectura Técnica**

### **🎨 Flujo de Evaluación Multi-Agente**

1. **Input**: Usuario ingresa `prompt` + `respuesta_del_LLM`
2. **Contexto**: Sistema crea `EvaluationContext` con ambos elementos
3. **Evaluación Paralela**: Los 5 jueces analizan simultáneamente en ~10-30 segundos
4. **Meta-Agregación**: Combina scores con pesos personalizados + análisis de consenso
5. **Output**: Score final + feedback detallado + visualización

### **🤖 Modelos de Hugging Face Utilizados**

#### **🎯 Dr. Precisión** (Peso: 35%)
- **Técnica**: Análisis heurístico inteligente + NLP
- **Evalúa**: Exactitud factual, ausencia de alucinaciones, referencias válidas
- **Algoritmos**:
  - Detecta indicadores de incertidumbre (`"quizás", "tal vez", "posiblemente"`)
  - Penaliza declaraciones absolutas (`"siempre", "nunca", "todos"`)
  - Busca números específicos, fechas, y citaciones científicas
  - Analiza la relación entre afirmaciones del prompt y respuesta

#### **🧠 Prof. Coherencia** (Peso: 30%)
- **Modelo HF**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Técnica**: Embeddings semánticos + similitud coseno
- **Evalúa**: Flujo lógico, transiciones, consistencia interna
- **Proceso**:
  1. Genera **embeddings semánticos** de cada oración
  2. Calcula **similitud coseno** entre oraciones consecutivas
  3. **Fallback**: Análisis de conectores lingüísticos si HF falla
  4. Considera cómo la respuesta mantiene coherencia con el prompt

#### **🎪 Lic. Relevancia** (Peso: 20%)
- **Modelo HF**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Técnica**: Similitud semántica prompt ↔ respuesta
- **Evalúa**: Pertinencia directa, completitud, foco temático
- **Proceso**:
  1. **Embeddings** separados de prompt y respuesta
  2. **Similitud semántica** entre ambos vectores
  3. **Fallback**: Análisis de overlap de palabras clave
  4. **Contexto**: Si el prompt hace una pregunta específica, ¿la respuesta la aborda?

#### **⚡ Ed. Eficiencia** (Peso: 10%)
- **Herramienta**: `textstat` library + métricas personalizadas
- **Técnica**: Índices de legibilidad + análisis de concisión
- **Evalúa**: Claridad, longitud apropiada, facilidad de lectura
- **Algoritmos**:
  - **Índice Flesch** de legibilidad (>60 = fácil de leer)
  - Análisis de longitud vs. complejidad del prompt
  - Detección si el prompt requiere respuesta detallada
  - Métricas de palabras por oración

#### **🎨 Dra. Creatividad** (Peso: 5%)
- **Técnica**: Análisis léxico-estadístico + heurísticas
- **Evalúa**: Originalidad, diversidad léxica, perspectivas únicas
- **Algoritmos**:
  - **Diversidad léxica**: unique_words / total_words
  - Detección de **indicadores creativos** (`"imaginemos", "supongamos", "metáfora"`)
  - Análisis de **variedad en estructura** de oraciones
  - Considera si el prompt solicita creatividad específicamente

### **🔄 Robustez y Modos Fallback**

El sistema tiene **tolerancia a fallos** incorporada:

- **🔄 Inicialización**: Timeout de 2 minutos, continua con fallbacks si HF falla
- **⚙️ Fallback Inteligente**: Si `SentenceTransformers` no carga → heurísticas de NLP
- **🛡️ Manejo de Errores**: Juez individual falla → score neutro (5.0)
- **📊 Consenso**: Analiza discrepancias entre jueces para detectar casos ambiguos

### **📊 Análisis de Consenso**

```python
# Mide qué tan de acuerdo están los 5 jueces
consensus_level = 1.0 - (std_dev_scores / 5.0)

# Interpretación:
# > 0.8 = Alto consenso (jueces de acuerdo)
# 0.6-0.8 = Consenso moderado (algunas diferencias)
# < 0.6 = Bajo consenso (evaluación compleja/ambigua)
```

### **🎯 Evaluación Contextual Prompt-Respuesta**

**Característica Clave**: El sistema NO evalúa respuestas en el vacío.

✅ **Lo que hace el sistema**:
- Analiza cómo la respuesta **responde específicamente** al prompt
- Considera si el prompt requiere **creatividad, precisión, o detalle**
- Evalúa la **relevancia semántica** entre pregunta y respuesta
- Ajusta expectativas según el **dominio** (técnico, creativo, académico)

❌ **Lo que NO hace**:
- Evaluar respuestas sin contexto del prompt original
- Aplicar criterios uniformes independientes de la pregunta
- Ignorar la intención y complejidad del prompt
