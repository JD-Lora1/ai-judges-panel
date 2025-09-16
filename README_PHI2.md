# AI Judges Panel - Phi-2 Implementation

🤖 **Single Model Architecture** powered by Microsoft Phi-2

## Overview

This is a streamlined version of the AI Judges Panel that uses a single, efficient Microsoft Phi-2 model (2.7B parameters) for comprehensive AI evaluation. The system has been completely refactored to focus on performance, simplicity, and resource efficiency.

## Key Features

✅ **Single Model**: Uses only Microsoft Phi-2 for all evaluations  
✅ **Resource Efficient**: Optimized memory usage and performance  
✅ **Comprehensive Evaluation**: Covers relevance, coherence, accuracy, and completeness  
✅ **Custom Weights**: Flexible importance weighting for different aspects  
✅ **Batch Processing**: Efficient handling of multiple evaluations  
✅ **Response Comparison**: Side-by-side evaluation of different responses  
✅ **Web Interface**: Modern FastAPI-based web application  
✅ **API Integration**: RESTful API for external integration  

## Architecture Changes

### What's New:
- **Single Phi-2 Judge**: Replaced multiple judge system with one powerful model
- **Simplified API**: Streamlined endpoints focused on core functionality  
- **Optimized Performance**: Singleton pattern with lazy loading and memory management
- **Clean Frontend**: Updated JavaScript for new API structure
- **Resource Management**: Built-in model loading/unloading capabilities

### What's Removed:
- Multiple LLM judge implementations
- Complex contextual judge systems
- Legacy test files and old implementations
- Heavy dependency requirements

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Implementation

```bash
python test_phi2.py
```

### 3. Run the Web Application

```bash
python -m app.main
```

or

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Try the Jupyter Notebook

```bash
jupyter notebook notebooks/phi2_evaluation_demo.ipynb
```

## API Endpoints

### Core Evaluation
- `POST /api/v1/evaluate` - Evaluate text using Phi-2
- `GET /api/v1/models/info` - Get Phi-2 model information

### Batch Operations  
- `POST /api/v1/evaluate/batch` - Batch evaluation
- `POST /api/v1/evaluate/compare` - Compare multiple responses

### Utility Endpoints
- `GET /api/v1/model/default-weights` - Get default evaluation weights
- `POST /api/v1/model/validate-weights` - Validate custom weights
- `GET /health` - Health check

## Usage Examples

### Basic Evaluation

```python
from app.models.phi2_judge import get_phi2_judge

judge = get_phi2_judge()
result = judge.evaluate(
    prompt="Explain quantum computing",
    response="Quantum computing uses quantum mechanics principles..."
)

print(f"Score: {result['overall_score']}/10")
```

### Custom Weights

```python
custom_weights = {
    "relevance": 0.4,
    "accuracy": 0.3,
    "coherence": 0.2,
    "completeness": 0.1
}

result = judge.evaluate(prompt, response, weights=custom_weights)
```

### Response Comparison

```python
result = judge.compare_responses(
    prompt="What is AI?",
    response1="AI is artificial intelligence...",
    response2="AI means smart computers..."
)

print(f"Winner: {result['winner']}")
```

## Web Interface

Access the web interface at `http://localhost:8000`:

- **Home** (`/`): Overview and quick start
- **Evaluate** (`/evaluate`): Interactive evaluation interface
- **About** (`/about`): System information

## Model Information

- **Model**: Microsoft Phi-2 (2.7B parameters)
- **Capabilities**: Text generation, reasoning, code understanding
- **Aspects Evaluated**: Relevance, Coherence, Accuracy, Completeness
- **Default Weights**: 30% relevance, 25% coherence, 25% accuracy, 20% completeness

## Performance Characteristics

- **Model Loading**: ~10-30 seconds (one-time, lazy loaded)
- **Evaluation Speed**: ~2-5 seconds per evaluation (varies by hardware)
- **Memory Usage**: ~6-8GB RAM (model + overhead)
- **GPU Support**: Automatic CUDA detection and usage

## Deployment

### Railway Deployment

The application is ready for Railway deployment:

1. Push to GitHub repository
2. Connect Railway to your repo
3. Deploy automatically

Configuration files included:
- `Procfile` - Railway process configuration
- `railway.toml` - Railway deployment settings
- `Dockerfile` - Container configuration (optional)

### Local Development

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## File Structure

```
ai-judges-panel/
├── app/
│   ├── api/
│   │   └── evaluation.py      # API routes
│   ├── models/
│   │   └── phi2_judge.py      # Phi-2 judge implementation
│   ├── static/
│   │   ├── css/main.css       # Styles
│   │   └── js/main.js         # Frontend JavaScript
│   ├── templates/             # HTML templates
│   └── main.py               # FastAPI application
├── notebooks/
│   └── phi2_evaluation_demo.ipynb  # Jupyter demo
├── test_phi2.py              # Test script
├── requirements.txt          # Dependencies
├── Procfile                  # Railway deployment
├── railway.toml             # Railway configuration
└── README_PHI2.md           # This file
```

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check internet connection for model download
   - Ensure sufficient RAM (8GB+ recommended)
   - Verify PyTorch installation

2. **Slow Performance**
   - Use GPU if available
   - Reduce evaluation batch sizes
   - Check system resources

3. **Import Errors**
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Performance Optimization

1. **Use GPU**: Automatic CUDA detection for faster inference
2. **Singleton Pattern**: Model loaded once and reused
3. **Memory Management**: Built-in model unloading capability
4. **Batch Processing**: Efficient handling of multiple evaluations

## Development

### Running Tests

```bash
python test_phi2.py
```

### Code Quality

The codebase follows clean architecture principles:
- Single responsibility per module
- Resource-efficient design
- Error handling and logging
- Type hints and documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support:
- Check the Jupyter notebook for examples
- Review the test script for usage patterns
- Examine the API documentation in the code

---

**Built with ❤️ using Microsoft Phi-2 and FastAPI**
