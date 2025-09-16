// AI Judges Panel - Main JavaScript

// Global variables
let isEvaluating = false;
let currentEvaluation = null;
let modelInfo = null;

// DOM elements
const evaluationForm = document.getElementById('evaluation-form');
const resultsContainer = document.getElementById('results-container');
const loadingSpinner = document.getElementById('loading-spinner');

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🏛️ AI Judges Panel initialized');
    
    // Initialize form if present
    if (evaluationForm) {
        initializeEvaluationForm();
    }
    
    // Initialize tooltips
    initializeTooltips();
    
    // Check system status
    checkSystemStatus();
    
    // Load model info
    loadModelInfo();
    
    // Initialize weight controls if present
    initializeWeightControls();
});

// Initialize evaluation form
function initializeEvaluationForm() {
    evaluationForm.addEventListener('submit', handleEvaluationSubmit);
    
    // Auto-resize textarea
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', autoResize);
        autoResize.call(textarea); // Initial resize
    });
}

// Handle evaluation form submission
async function handleEvaluationSubmit(event) {
    event.preventDefault();
    
    if (isEvaluating) return;
    
    const formData = new FormData(evaluationForm);
    const evaluationData = {
        prompt: formData.get('prompt'),
        response: formData.get('response'),
        custom_weights: getCustomWeights()
    };
    
    // Validate input
    if (!evaluationData.prompt || !evaluationData.response) {
        showAlert('error', 'Por favor, completa tanto el prompt como la respuesta.');
        return;
    }
    
    if (evaluationData.prompt.length < 1) {
        showAlert('warning', 'El prompt no puede estar vacío.');
        return;
    }
    
    if (evaluationData.response.length < 1) {
        showAlert('warning', 'La respuesta no puede estar vacía.');
        return;
    }
    
    // Start evaluation
    await performEvaluation(evaluationData);
}

// Perform evaluation
async function performEvaluation(data) {
    isEvaluating = true;
    showLoadingState();
    hideResults();
    
    try {
        const response = await fetch('/api/v1/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Error ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        currentEvaluation = result;
        
        // Display results with animation delay
        setTimeout(() => {
            displayResults(result);
        }, 500);
        
    } catch (error) {
        console.error('Evaluation error:', error);
        showAlert('error', `Error en la evaluación: ${error.message}`);
    } finally {
        isEvaluating = false;
        hideLoadingState();
    }
}

// Display evaluation results
function displayResults(evaluation) {
    if (!resultsContainer) return;
    
    const overallScore = evaluation.overall_score;
    const scoreClass = getScoreClass(overallScore);
    
    resultsContainer.innerHTML = `
        <div class="results-header text-center mb-4">
            <h3><i class="fas fa-chart-line me-2"></i>Resultados de la Evaluación Phi-2</h3>
            <div class="score-display ${scoreClass}">
                ${overallScore.toFixed(1)}/10
            </div>
            <div class="consensus-info">
                <span class="badge bg-info">
                    <i class="fas fa-robot me-1"></i>
                    Phi-2 Model
                </span>
                <span class="badge bg-secondary ms-2">
                    <i class="fas fa-clock me-1"></i>
                    ${evaluation.evaluation_time.toFixed(2)}s
                </span>
                <span class="badge bg-success ms-2">
                    <i class="fas fa-microchip me-1"></i>
                    ${evaluation.model_info.device}
                </span>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6 mb-4">
                <h5><i class="fas fa-chart-bar me-2"></i>Scores por Aspecto</h5>
                <div class="aspect-scores">
                    ${generateAspectScores(evaluation.detailed_scores)}
                </div>
            </div>
            
            <div class="col-md-6 mb-4">
                <h5><i class="fas fa-comment-alt me-2"></i>Retroalimentación Detallada</h5>
                <div class="detailed-feedback">
                    ${generateDetailedFeedback(evaluation.detailed_feedback)}
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6 mb-4">
                <h5><i class="fas fa-balance-scale me-2"></i>Pesos Utilizados</h5>
                <div class="weights-info">
                    ${generateWeightsDisplay(evaluation.weights_used)}
                </div>
            </div>
            
            <div class="col-md-6 mb-4">
                <h5><i class="fas fa-info-circle me-2"></i>Información del Modelo</h5>
                <div class="model-info">
                    <small class="text-muted">
                        <strong>Modelo:</strong> ${evaluation.model_info.name}<br>
                        <strong>Parámetros:</strong> ${evaluation.model_info.model_parameters}<br>
                        <strong>Dispositivo:</strong> ${evaluation.model_info.device}<br>
                        <strong>Prompt:</strong> ${evaluation.input_info.prompt_length} chars<br>
                        <strong>Respuesta:</strong> ${evaluation.input_info.response_length} chars<br>
                        <strong>Timestamp:</strong> ${new Date(evaluation.timestamp).toLocaleString()}
                    </small>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="chart-container">
                    <canvas id="scoresChart" width="400" height="200"></canvas>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-12 text-center">
                <button class="btn btn-primary me-2" onclick="downloadResults()">
                    <i class="fas fa-download me-1"></i>Descargar Resultados
                </button>
                <button class="btn btn-outline-secondary me-2" onclick="shareResults()">
                    <i class="fas fa-share me-1"></i>Compartir
                </button>
                <button class="btn btn-outline-info" onclick="showDetailedAnalysis()">
                    <i class="fas fa-microscope me-1"></i>Análisis Detallado
                </button>
            </div>
        </div>
    `;
    
    // Show results with animation
    resultsContainer.style.display = 'block';
    resultsContainer.classList.add('fade-in');
    
    // Create scores chart
    createScoresChart(evaluation.detailed_scores);
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Generate aspect scores HTML
function generateAspectScores(scores) {
    const aspectIcons = {
        'precision': '🎯',
        'coherence': '🧠',
        'relevance': '🎪',
        'efficiency': '⚡',
        'creativity': '🎨'
    };
    
    return Object.entries(scores).map(([aspect, score]) => {
        const percentage = (score / 10) * 100;
        const progressClass = getProgressClass(score);
        
        return `
            <div class="aspect-score mb-3">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span>
                        ${aspectIcons[aspect] || '📊'} ${aspect.charAt(0).toUpperCase() + aspect.slice(1)}
                    </span>
                    <strong>${score.toFixed(1)}/10</strong>
                </div>
                <div class="progress">
                    <div class="progress-bar ${progressClass}" 
                         role="progressbar" 
                         style="width: ${percentage}%"
                         aria-valuenow="${score}" 
                         aria-valuemin="0" 
                         aria-valuemax="10">
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Create scores chart
function createScoresChart(scores) {
    const ctx = document.getElementById('scoresChart');
    if (!ctx) return;
    
    const labels = Object.keys(scores).map(aspect => 
        aspect.charAt(0).toUpperCase() + aspect.slice(1)
    );
    const data = Object.values(scores);
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Puntuación',
                data: data,
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(102, 126, 234, 1)'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 10,
                    ticks: {
                        stepSize: 2
                    }
                }
            }
        }
    });
}

// Utility functions
function getScoreClass(score) {
    if (score >= 8) return 'score-excellent';
    if (score >= 6) return 'score-good';
    if (score >= 4) return 'score-average';
    return 'score-poor';
}

function getProgressClass(score) {
    if (score >= 8) return 'bg-success';
    if (score >= 6) return 'bg-info';
    if (score >= 4) return 'bg-warning';
    return 'bg-danger';
}

function showLoadingState() {
    if (loadingSpinner) {
        loadingSpinner.style.display = 'block';
    }
    
    const submitButton = document.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="loading-spinner me-2"></span>Evaluando...';
    }
}

function hideLoadingState() {
    if (loadingSpinner) {
        loadingSpinner.style.display = 'none';
    }
    
    const submitButton = document.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="fas fa-play me-2"></i>Evaluar Respuesta';
    }
}

function showResults() {
    if (resultsContainer) {
        resultsContainer.style.display = 'block';
    }
}

function hideResults() {
    if (resultsContainer) {
        resultsContainer.style.display = 'none';
        resultsContainer.classList.remove('fade-in');
    }
}

function showAlert(type, message) {
    const alertClass = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    }[type] || 'alert-info';
    
    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Insert alert at top of main content
    const main = document.querySelector('main');
    if (main) {
        main.insertAdjacentHTML('afterbegin', alertHtml);
    }
}

function autoResize() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/v1/models/available');
        const data = await response.json();
        availableModels = data.available_models;
        
        // Update model selector if present
        const modelSelector = document.getElementById('llm_model');
        if (modelSelector && availableModels.length > 0) {
            modelSelector.innerHTML = availableModels.map(model => 
                `<option value="${model.id}">${model.name}</option>`
            ).join('');
        }
        
    } catch (error) {
        console.warn('Could not load available models:', error);
    }
}

async function checkSystemStatus() {
    try {
        const response = await fetch('/health');
        const status = await response.json();
        
        // Update status indicators if present
        const statusElements = document.querySelectorAll('[data-status]');
        statusElements.forEach(element => {
            const statusType = element.dataset.status;
            if (statusType === 'system' && status.status === 'healthy') {
                element.className = 'badge bg-success';
                element.textContent = 'Activo';
            }
        });
        
    } catch (error) {
        console.warn('Could not check system status:', error);
    }
}

// Action functions
function downloadResults() {
    if (!currentEvaluation) return;
    
    const dataStr = JSON.stringify(currentEvaluation, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `ai-judges-evaluation-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    showAlert('success', 'Resultados descargados correctamente');
}

function shareResults() {
    if (!currentEvaluation) return;
    
    const shareText = `AI Judges Panel - Score: ${currentEvaluation.final_score.toFixed(1)}/10 (Consenso: ${(currentEvaluation.consensus_level * 100).toFixed(0)}%)`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Evaluación AI Judges Panel',
            text: shareText,
            url: window.location.href
        });
    } else {
        // Fallback to clipboard
        navigator.clipboard.writeText(shareText + ' - ' + window.location.href)
            .then(() => showAlert('success', 'Enlace copiado al portapapeles'))
            .catch(() => showAlert('error', 'No se pudo copiar al portapapeles'));
    }
}

function showDetailedAnalysis() {
    if (!currentEvaluation) return;
    
    // Create modal with detailed analysis
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-microscope me-2"></i>Análisis Detallado
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="row">
                        <div class="col-12">
                            <h6>Feedback Completo por Juez:</h6>
                            ${generateDetailedFeedback(currentEvaluation.detailed_feedback)}
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cerrar</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
    
    // Remove modal from DOM when hidden
    modal.addEventListener('hidden.bs.modal', () => {
        document.body.removeChild(modal);
    });
}

function generateDetailedFeedback(detailedFeedback) {
    if (!detailedFeedback) return '<p>No hay feedback detallado disponible.</p>';
    
    return Object.entries(detailedFeedback).map(([aspect, feedback]) => `
        <div class="card mb-3">
            <div class="card-header">
                <strong>${aspect.charAt(0).toUpperCase() + aspect.slice(1)}</strong>
            </div>
            <div class="card-body">
                <p>${feedback}</p>
            </div>
        </div>
    `).join('');
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/api/v1/models/info');
        if (response.ok) {
            modelInfo = await response.json();
            updateModelDisplay();
        }
    } catch (error) {
        console.warn('Could not load model info:', error);
    }
}

// Update model display
function updateModelDisplay() {
    const modelElements = document.querySelectorAll('[data-model-info]');
    modelElements.forEach(element => {
        const infoType = element.dataset.modelInfo;
        if (modelInfo && modelInfo.model_info) {
            switch(infoType) {
                case 'name':
                    element.textContent = modelInfo.model_info.model_name;
                    break;
                case 'device':
                    element.textContent = modelInfo.model_info.device;
                    break;
                case 'status':
                    element.textContent = modelInfo.status;
                    break;
            }
        }
    });
}

// Get custom weights from form
function getCustomWeights() {
    const weightInputs = document.querySelectorAll('[data-aspect-weight]');
    if (weightInputs.length === 0) return null;
    
    const weights = {};
    weightInputs.forEach(input => {
        const aspect = input.dataset.aspectWeight;
        const value = parseFloat(input.value);
        if (!isNaN(value)) {
            weights[aspect] = value;
        }
    });
    
    return Object.keys(weights).length > 0 ? weights : null;
}

// Initialize weight controls
function initializeWeightControls() {
    const weightContainer = document.getElementById('weight-controls');
    if (!weightContainer) return;
    
    // Create weight sliders for each aspect
    const aspects = ['relevance', 'coherence', 'accuracy', 'completeness'];
    const defaultWeights = [0.3, 0.25, 0.25, 0.2];
    
    weightContainer.innerHTML = aspects.map((aspect, index) => `
        <div class="mb-3">
            <label for="weight-${aspect}" class="form-label">
                ${aspect.charAt(0).toUpperCase() + aspect.slice(1)}
                <span class="badge bg-secondary" id="weight-${aspect}-display">${defaultWeights[index]}</span>
            </label>
            <input type="range" class="form-range" 
                   id="weight-${aspect}" 
                   data-aspect-weight="${aspect}"
                   min="0" max="1" step="0.05" value="${defaultWeights[index]}">
        </div>
    `).join('');
    
    // Add event listeners for real-time updates
    aspects.forEach(aspect => {
        const slider = document.getElementById(`weight-${aspect}`);
        const display = document.getElementById(`weight-${aspect}-display`);
        
        slider.addEventListener('input', function() {
            display.textContent = this.value;
        });
    });
}

// Generate weights display
function generateWeightsDisplay(weights) {
    return Object.entries(weights).map(([aspect, weight]) => {
        const percentage = (weight * 100).toFixed(1);
        return `
            <div class="d-flex justify-content-between mb-2">
                <span>${aspect.charAt(0).toUpperCase() + aspect.slice(1)}:</span>
                <span class="badge bg-primary">${percentage}%</span>
            </div>
        `;
    }).join('');
}

// Update share results for Phi-2
function shareResults() {
    if (!currentEvaluation) return;
    
    const shareText = `AI Judges Panel (Phi-2) - Score: ${currentEvaluation.overall_score.toFixed(1)}/10`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Evaluación AI Judges Panel - Phi-2',
            text: shareText,
            url: window.location.href
        });
    } else {
        navigator.clipboard.writeText(shareText + ' - ' + window.location.href)
            .then(() => showAlert('success', 'Enlace copiado al portapapeles'))
            .catch(() => showAlert('error', 'No se pudo copiar al portapapeles'));
    }
}

// Export functions for use in other scripts
window.AIJudgesPanel = {
    performEvaluation,
    showAlert,
    downloadResults,
    shareResults,
    showDetailedAnalysis,
    loadModelInfo,
    getCustomWeights
};
