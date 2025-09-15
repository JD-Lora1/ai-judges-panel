# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**AI Judges Panel** is a multi-agent architecture for LLM evaluation deployed as a FastAPI web application. Specialized AI judges powered by Hugging Face models evaluate responses from different perspectives (precision, creativity, coherence, relevance, efficiency). The system provides both a web interface and REST API for comprehensive, interpretable assessments.

**Key Features:**
- 🌐 **Web Application**: Complete FastAPI interface with interactive evaluation
- 🤖 **Hugging Face Integration**: Real models (BERT, SentenceTransformers) replace simulations
- 🚀 **Railway Deployment**: Ready-to-deploy configuration for cloud hosting
- 📊 **REST API**: Full API with batch processing, comparison, and management endpoints
- 📱 **Responsive UI**: Bootstrap-based interface with real-time status and visualizations

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup Jupyter for notebooks
jupyter notebook

# Development dependencies
pip install pytest black python-dotenv
```

### Running the Project
```bash
# Run web application (primary interface)
uvicorn app.main:app --reload
# Access at: http://localhost:8000

# Run with specific port
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Start Jupyter notebooks for research (optional)
jupyter notebook notebooks/

# Quick test of core functionality
python -c "
import asyncio
from app.models.hf_judges import HuggingFaceJudgesPanel

async def test():
    panel = HuggingFaceJudgesPanel()
    await panel.initialize()
    result = await panel.evaluate('Test prompt', 'Test response')
    print(f'Score: {result.final_score:.1f}/10')
    
asyncio.run(test())
"
```

### Testing and Code Quality
```bash
# Run tests (when implemented)
pytest tests/

# Format code with black
black src/ --line-length 100

# Check Python syntax
python -m py_compile src/**/*.py
```

## Architecture Overview

### Core Components

1. **Judge System** (`src/judges/`):
   - `BaseJudge`: Abstract base class defining the judge interface
   - `PrecisionJudge`: Evaluates factual accuracy and detects hallucinations
   - Specialized judges for creativity, coherence, relevance, and efficiency (planned/simulated)

2. **Meta-Evaluator** (`src/evaluators/`):
   - Orchestrates multiple judges and combines their evaluations
   - Calculates consensus levels and weighted final scores
   - Provides comprehensive evaluation reports with detailed feedback

3. **Data Structures**:
   - `JudgeEvaluation`: Individual judge assessment with score, reasoning, and metadata
   - `ComprehensiveEvaluation`: Complete panel evaluation with consensus analysis
   - `EvaluationContext`: Input context containing prompts and responses

### Key Design Patterns

- **Strategy Pattern**: Each judge implements the same interface but with specialized evaluation logic
- **Aggregation Pattern**: Meta-evaluator combines multiple judge opinions with configurable weights
- **Simulation Layer**: Current implementation uses simulated LLM responses for demonstration

### Module Import Structure

The project requires manual path management since there are no `__init__.py` files:

```python
import sys
import os
sys.path.append(os.path.abspath('src'))  # From notebooks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # From src/
```

## Development Notes

### Current Implementation Status

- **Fully Implemented**: `PrecisionJudge` with detailed evaluation logic
- **Simulated**: Other judges (creativity, coherence, relevance, efficiency) use placeholder implementations
- **Demo Ready**: Complete end-to-end evaluation pipeline with visualization

### Extension Points

1. **Adding New Judges**: Inherit from `BaseJudge` and implement the three abstract methods
2. **Real LLM Integration**: Replace `_call_llm` methods with actual API calls (OpenAI, Anthropic, etc.)
3. **Automatic Metrics**: Implement BLEU, ROUGE, BERTScore in `meta_evaluator._calculate_automatic_metrics`

### Working with Notebooks

The primary interface is through Jupyter notebooks in `notebooks/`:
- Start with `01_demo_basic.ipynb` for complete system overview
- Notebooks handle path setup and provide interactive demonstrations
- All visualizations and analysis tools are notebook-based

### Data Flow

1. **Input**: Original prompt + candidate response → `EvaluationContext`
2. **Individual Evaluation**: Each judge analyzes the response → `JudgeEvaluation`
3. **Aggregation**: Meta-evaluator combines judge scores → `ComprehensiveEvaluation`
4. **Output**: Final score + detailed feedback + consensus analysis

### Dependencies and Environment

- **Core**: pandas, numpy, matplotlib, seaborn for data handling and visualization
- **NLP**: nltk, rouge-score for text metrics (with optional advanced metrics commented)
- **Development**: jupyter, pytest, black for development workflow
- **Optional**: OpenAI/Anthropic APIs for real LLM integration (commented in requirements)

### Configuration

- Weights are configurable in `MetaEvaluator` initialization
- Judge parameters (temperature, model) are customizable per judge
- No external configuration files - all settings are code-based

## Project-Specific Guidelines

### Working with Judges

Each judge must implement three methods:
- `_generate_system_prompt()`: Define judge personality and expertise
- `_generate_evaluation_prompt(context)`: Create evaluation-specific prompts  
- `_parse_llm_response(response)`: Convert LLM output to structured evaluation

### Evaluation Context

Always provide meaningful context when evaluating:
- Include domain information when available
- Use reference responses for automatic metrics
- Consider task type for specialized evaluation

### Error Handling

The system includes robust fallback mechanisms:
- Failed judge evaluations default to neutral scores (5.0)
- Parsing errors trigger regex-based extraction
- Comprehensive error metadata for debugging
