// EvolvAttention Web Interface
// Comprehensive JavaScript for API interaction and visualization

class EvolvAttentionUI {
    constructor() {
        this.apiBase = '/api/v1';
        this.evolutionActive = false;
        this.currentGeneration = 0;
        this.evolutionHistory = [];
        this.attentionData = null;
        this.similarityData = null;
        
        this.initializeEventListeners();
        this.checkServerHealth();
    }

    // Initialize all event listeners
    initializeEventListeners() {
        // Barycenter calculation
        document.getElementById('calculate-barycenter').addEventListener('click', () => {
            this.calculateBarycenter();
        });

        // Cosine similarities
        document.getElementById('calculate-cosine').addEventListener('click', () => {
            this.calculateCosineSimilarities();
        });

        // Evolution controls
        document.getElementById('initialize-evolution').addEventListener('click', () => {
            this.initializeEvolution();
        });

        document.getElementById('step-evolution').addEventListener('click', () => {
            this.stepEvolution();
        });

        document.getElementById('reset-evolution').addEventListener('click', () => {
            this.resetEvolution();
        });

        // Attention analysis
        document.getElementById('analyze-attention').addEventListener('click', () => {
            this.analyzeAttention();
        });

        // Visualization controls
        document.getElementById('show-similarity-chart').addEventListener('click', () => {
            this.showSimilarityChart();
        });

        document.getElementById('show-evolution-progress').addEventListener('click', () => {
            this.showEvolutionProgress();
        });

        document.getElementById('show-attention-heatmap').addEventListener('click', () => {
            this.showAttentionHeatmap();
        });
    }

    // Check server health
    async checkServerHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            if (data.status === 'healthy') {
                this.showStatus('Server is running and healthy', 'success');
            }
        } catch (error) {
            this.showStatus('Server connection failed', 'error');
        }
    }

    // Utility functions
    showStatus(message, type = 'info') {
        const statusElements = document.querySelectorAll('.status');
        statusElements.forEach(element => {
            element.textContent = message;
            element.className = `status ${type}`;
        });
    }

    showLoading(button) {
        const originalText = button.textContent;
        button.innerHTML = '<span class="loading"></span>Loading...';
        button.disabled = true;
        return () => {
            button.textContent = originalText;
            button.disabled = false;
        };
    }

    parseTextarea(textareaId) {
        const textarea = document.getElementById(textareaId);
        return textarea.value.split('\n').filter(line => line.trim() !== '');
    }

    // API call wrapper
    async apiCall(endpoint, method = 'GET', data = null) {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(`${this.apiBase}${endpoint}`, options);
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || `HTTP ${response.status}`);
            }
            
            return result;
        } catch (error) {
            throw new Error(`API call failed: ${error.message}`);
        }
    }

    // Barycenter calculation
    async calculateBarycenter() {
        const strings = this.parseTextarea('barycenter-strings');
        if (strings.length === 0) {
            this.showStatus('Please enter at least one string', 'error');
            return;
        }

        const button = document.getElementById('calculate-barycenter');
        const stopLoading = this.showLoading(button);

        try {
            await this.apiCall('/barycenter', 'POST', { strings });
            this.showStatus('Barycenter calculated successfully', 'success');
        } catch (error) {
            this.showStatus(`Barycenter calculation failed: ${error.message}`, 'error');
        } finally {
            stopLoading();
        }
    }

    // Cosine similarities calculation
    async calculateCosineSimilarities() {
        const strings = this.parseTextarea('cosine-strings');
        if (strings.length === 0) {
            this.showStatus('Please enter at least one string', 'error');
            return;
        }

        const button = document.getElementById('calculate-cosine');
        const stopLoading = this.showLoading(button);

        try {
            const result = await this.apiCall('/cosine-similarities', 'POST', { strings });
            this.similarityData = { strings, similarities: result.data.similarities };
            this.displayCosineResults();
            this.showStatus('Cosine similarities calculated successfully', 'success');
        } catch (error) {
            this.showStatus(`Cosine calculation failed: ${error.message}`, 'error');
        } finally {
            stopLoading();
        }
    }

    // Display cosine similarity results
    displayCosineResults() {
        const resultsDiv = document.getElementById('cosine-results');
        resultsDiv.innerHTML = '';

        this.similarityData.strings.forEach((string, index) => {
            const similarity = this.similarityData.similarities[index];
            const percentage = Math.round(similarity * 100);
            
            const barElement = document.createElement('div');
            barElement.className = 'similarity-bar';
            barElement.innerHTML = `
                <div class="similarity-text">${string}</div>
                <div class="similarity-score">${similarity.toFixed(3)}</div>
                <div class="similarity-bar-fill" style="width: ${percentage}%"></div>
            `;
            resultsDiv.appendChild(barElement);
        });
    }

    // Initialize evolution
    async initializeEvolution() {
        const targetStrings = this.parseTextarea('target-strings');
        if (targetStrings.length === 0) {
            this.showStatus('Please enter at least one target string', 'error');
            return;
        }

        const populationSize = parseInt(document.getElementById('population-size').value);
        const stepGenerations = parseInt(document.getElementById('step-generations').value);
        const outputLength = parseInt(document.getElementById('output-length').value);

        const button = document.getElementById('initialize-evolution');
        const stopLoading = this.showLoading(button);

        try {
            const result = await this.apiCall('/evolution/initialize', 'POST', {
                target_strings: targetStrings,
                population_size: populationSize,
                step_generations: stepGenerations,
                output_length: outputLength
            });

            this.evolutionActive = true;
            this.currentGeneration = 0;
            this.evolutionHistory = [result.data];
            
            document.getElementById('step-evolution').disabled = false;
            document.getElementById('reset-evolution').disabled = false;
            
            this.displayEvolutionResults(result.data);
            this.showStatus('Evolution initialized successfully', 'success');
        } catch (error) {
            this.showStatus(`Evolution initialization failed: ${error.message}`, 'error');
        } finally {
            stopLoading();
        }
    }

    // Step evolution
    async stepEvolution() {
        if (!this.evolutionActive) {
            this.showStatus('No active evolution session', 'error');
            return;
        }

        const button = document.getElementById('step-evolution');
        const stopLoading = this.showLoading(button);

        try {
            const result = await this.apiCall('/evolution/step', 'POST', {});
            
            this.currentGeneration += result.data.generation;
            this.evolutionHistory.push(result.data);
            
            this.displayEvolutionResults(result.data);
            this.showStatus(`Evolution step completed. Generation: ${this.currentGeneration}`, 'success');
        } catch (error) {
            this.showStatus(`Evolution step failed: ${error.message}`, 'error');
        } finally {
            stopLoading();
        }
    }

    // Reset evolution
    resetEvolution() {
        this.evolutionActive = false;
        this.currentGeneration = 0;
        this.evolutionHistory = [];
        
        document.getElementById('step-evolution').disabled = true;
        document.getElementById('reset-evolution').disabled = true;
        
        document.getElementById('evolution-results').innerHTML = '';
        this.showStatus('Evolution reset', 'success');
    }

    // Display evolution results
    displayEvolutionResults(data) {
        const resultsDiv = document.getElementById('evolution-results');
        resultsDiv.innerHTML = `
            <h3>Generation ${data.generation || 0}</h3>
            <p><strong>Best Fitness:</strong> ${data.best_fitness?.toFixed(3) || 'N/A'}</p>
            <p><strong>Average Fitness:</strong> ${data.average_fitness?.toFixed(3) || 'N/A'}</p>
            <p><strong>Median Fitness:</strong> ${data.median_fitness?.toFixed(3) || 'N/A'}</p>
            <h4>Top Population Members:</h4>
        `;

        if (data.population) {
            data.population.slice(0, 10).forEach((item, index) => {
                const populationItem = document.createElement('div');
                populationItem.className = 'population-item';
                populationItem.innerHTML = `
                    <span class="fitness">${item.similarity.toFixed(3)}</span>
                    ${item.string}
                `;
                resultsDiv.appendChild(populationItem);
            });
        }
    }

    // Analyze attention
    async analyzeAttention() {
        const string = document.getElementById('attention-string').value.trim();
        if (!string) {
            this.showStatus('Please enter a string to analyze', 'error');
            return;
        }

        const button = document.getElementById('analyze-attention');
        const stopLoading = this.showLoading(button);

        try {
            const result = await this.apiCall('/attention/analyze', 'POST', { string });
            this.attentionData = result.data;
            this.displayAttentionResults();
            this.showStatus('Attention analysis completed successfully', 'success');
        } catch (error) {
            this.showStatus(`Attention analysis failed: ${error.message}`, 'error');
        } finally {
            stopLoading();
        }
    }

    // Display attention results
    displayAttentionResults() {
        const resultsDiv = document.getElementById('attention-results');
        resultsDiv.innerHTML = `
            <h3>Attention Analysis Results</h3>
            <p><strong>Overall Score:</strong> ${this.attentionData.overall_score.toFixed(3)}</p>
            <h4>Component Analysis:</h4>
        `;

        if (this.attentionData.components) {
            this.attentionData.components.forEach(component => {
                const score = component.score;
                const intensity = Math.round(score * 255);
                const color = `rgb(${intensity}, ${255 - intensity}, 100)`;
                
                const componentElement = document.createElement('span');
                componentElement.className = 'attention-component';
                componentElement.style.backgroundColor = color;
                componentElement.textContent = component.text;
                componentElement.title = `Score: ${score.toFixed(3)}`;
                
                resultsDiv.appendChild(componentElement);
            });
        }
    }

    // Visualization functions
    showSimilarityChart() {
        if (!this.similarityData) {
            this.showStatus('No similarity data available. Calculate similarities first.', 'error');
            return;
        }

        const vizArea = document.getElementById('visualization-area');
        vizArea.innerHTML = '';
        vizArea.className = 'visualization-area has-content';

        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';

        this.similarityData.strings.forEach((string, index) => {
            const similarity = this.similarityData.similarities[index];
            const height = similarity * 200; // Scale to chart height
            
            const bar = document.createElement('div');
            bar.className = 'chart-bar';
            bar.style.height = `${height}px`;
            bar.style.width = `${100 / this.similarityData.strings.length - 2}%`;
            bar.title = `${string}: ${similarity.toFixed(3)}`;
            
            chartContainer.appendChild(bar);
        });

        vizArea.appendChild(chartContainer);
    }

    showEvolutionProgress() {
        if (this.evolutionHistory.length === 0) {
            this.showStatus('No evolution data available. Initialize evolution first.', 'error');
            return;
        }

        const vizArea = document.getElementById('visualization-area');
        vizArea.innerHTML = '';
        vizArea.className = 'visualization-area has-content';

        const progressDiv = document.createElement('div');
        progressDiv.innerHTML = `
            <h3>Evolution Progress</h3>
            <p>Total Generations: ${this.currentGeneration}</p>
            <p>Best Fitness History:</p>
        `;

        const fitnessHistory = this.evolutionHistory.map(h => h.best_fitness).filter(f => f !== undefined);
        if (fitnessHistory.length > 0) {
            const chartContainer = document.createElement('div');
            chartContainer.className = 'chart-container';
            chartContainer.style.display = 'flex';
            chartContainer.style.alignItems = 'flex-end';
            chartContainer.style.gap = '2px';

            fitnessHistory.forEach((fitness, index) => {
                const height = fitness * 200;
                const bar = document.createElement('div');
                bar.className = 'chart-bar';
                bar.style.height = `${height}px`;
                bar.style.flex = '1';
                bar.title = `Generation ${index * 10}: ${fitness.toFixed(3)}`;
                chartContainer.appendChild(bar);
            });

            progressDiv.appendChild(chartContainer);
        }

        vizArea.appendChild(progressDiv);
    }

    showAttentionHeatmap() {
        if (!this.attentionData) {
            this.showStatus('No attention data available. Analyze attention first.', 'error');
            return;
        }

        const vizArea = document.getElementById('visualization-area');
        vizArea.innerHTML = '';
        vizArea.className = 'visualization-area has-content';

        const heatmapDiv = document.createElement('div');
        heatmapDiv.innerHTML = '<h3>Attention Heatmap</h3>';
        
        const heatmap = document.createElement('div');
        heatmap.className = 'heatmap';

        if (this.attentionData.components) {
            this.attentionData.components.forEach(component => {
                const score = component.score;
                const intensity = Math.round(score * 255);
                const color = `rgb(${intensity}, ${255 - intensity}, 100)`;
                
                const cell = document.createElement('div');
                cell.className = 'heatmap-cell';
                cell.style.backgroundColor = color;
                cell.textContent = component.text.charAt(0).toUpperCase();
                cell.title = `${component.text}: ${score.toFixed(3)}`;
                
                heatmap.appendChild(cell);
            });
        }

        heatmapDiv.appendChild(heatmap);
        vizArea.appendChild(heatmapDiv);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new EvolvAttentionUI();
});
