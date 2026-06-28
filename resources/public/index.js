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
        
        this.initializeNav();
        this.initializeEventListeners();
        this.checkServerHealth();
    }

    // Switch between the top-level nav views
    initializeNav() {
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', (event) => {
                event.preventDefault();
                const view = item.dataset.view;

                navItems.forEach(n => n.classList.toggle('active', n === item));
                document.querySelectorAll('.view').forEach(v => {
                    v.classList.toggle('active', v.id === `${view}-view`);
                });
            });
        });
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
        const repetitionDiscount = document.getElementById('repetition-discount').checked;
        const maxTarget = document.getElementById('max-target').checked;

        const button = document.getElementById('initialize-evolution');
        const stopLoading = this.showLoading(button);

        try {
            const result = await this.apiCall('/evolution/initialize', 'POST', {
                target_strings: targetStrings,
                population_size: populationSize,
                step_generations: stepGenerations,
                output_length: outputLength,
                genome_mode: 'word',
                repetition_discount: repetitionDiscount,
                target_mode: maxTarget ? 'max' : 'barycenter'
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
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new EvolvAttentionUI();
});
