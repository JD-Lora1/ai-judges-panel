// AI Judges Panel - Main JavaScript

// Global variables
let isEvaluating = false;
let currentEvaluation = null;
let availableModels = [];
let currentModelType = 'hf';

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
    
    // Load available models
    loadAvailableModels();
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
        domain: formData.get('domain') || null,
        include_automatic_metrics: formData.get('include_automatic_metrics') === 'on',
        model_type: formData.get('model_type') || 'hf',
        llm_model: formData.get('llm_model') || 'google/flan-t5-base'
    };
    
    // Validate input
    if (!evaluationData.prompt || !evaluationData.response) {
        showAlert('error', 'Por favor, completa tanto el prompt como la respuesta.');
        return;
    }
    
    if (evaluationData.prompt.length < 10) {
        showAlert('warning', 'El prompt debe tener al menos 10 caracteres.');
        return;
    }
    
    if (evaluationData.response.length < 10) {
        showAlert('warning', 'La respuesta debe tener al menos 10 caracteres.');
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
        const response = await fetch('/api/v1/evaluate/detailed', {
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
    
    const finalScore = evaluation.final_score;
    const scoreClass = getScoreClass(finalScore);
    
    resultsContainer.innerHTML = `
        <div class="results-header text-center mb-4">
            <h3><i class="fas fa-chart-line me-2"></i>Resultados de la Evaluación</h3>
            <div class="score-display ${scoreClass}">
                ${finalScore.toFixed(1)}/10
            </div>
            <div class="consensus-info">
                <span class="badge bg-info">
                    <i class="fas fa-handshake me-1"></i>
                    Consenso: ${(evaluation.consensus_level * 100).toFixed(0)}%
                </span>
                <span class="badge bg-secondary ms-2">
                    <i class="fas fa-clock me-1"></i>
                    ${evaluation.evaluation_time.toFixed(2)}s
                </span>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6 mb-4">
                <h5><i class="fas fa-chart-bar me-2"></i>Scores por Aspecto</h5>
                <div class="aspect-scores">
                    ${generateAspectScores(evaluation.individual_scores)}
                </div>
            </div>
            
            <div class="col-md-6 mb-4">
                <h5><i class="fas fa-star me-2"></i>Fortalezas</h5>
                <ul class="feedback-list strengths">
                    ${evaluation.strengths.slice(0, 5).map(strength => 
                        `<li>${strength}</li>`
                    ).join('')}
                </ul>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6 mb-4">
                <h5><i class="fas fa-arrow-up me-2"></i>Áreas de Mejora</h5>
                <ul class="feedback-list improvements">
                    ${evaluation.improvements.slice(0, 5).map(improvement => 
                        `<li>${improvement}</li>`
                    ).join('')}
                </ul>
            </div>
            
            <div class="col-md-6 mb-4">
                <h5><i class="fas fa-info-circle me-2"></i>Metadatos</h5>
                <div class="metadata-info">
                    <small class="text-muted">
                        <strong>Tipo:</strong> ${evaluation.metadata.model_type || 'hf'}<br>
                        ${evaluation.metadata.model_used ? `<strong>Modelo:</strong> ${evaluation.metadata.model_used}<br>` : ''}
                        ${evaluation.metadata.domain ? `<strong>Dominio:</strong> ${evaluation.metadata.domain}<br>` : ''}
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
    createScoresChart(evaluation.individual_scores);
    
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
                <span class="badge bg-primary ms-2">${feedback.score}/10</span>
            </div>
            <div class="card-body">
                <p><strong>Razonamiento:</strong> ${feedback.reasoning}</p>
                <div class="row">
                    <div class="col-md-6">
                        <h6>Fortalezas:</h6>
                        <ul class="small">
                            ${feedback.strengths.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>Mejoras:</h6>
                        <ul class="small">
                            ${feedback.improvements.map(i => `<li>${i}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

// Export functions for use in other scripts
window.AIJudgesPanel = {
    performEvaluation,
    showAlert,
    downloadResults,
    shareResults,
    showDetailedAnalysis
};
