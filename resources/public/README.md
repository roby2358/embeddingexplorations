# EvolvAttention Web Interface

A comprehensive web interface for the EvolvAttention API, providing interactive tools for evolutionary attention analysis.

## Features

### 1. Barycenter Calculation
- Input multiple strings to calculate their cosine barycenter
- Stores the barycenter in server memory for subsequent operations
- Essential first step for other analysis features

### 2. Cosine Similarities
- Compare test strings against the stored barycenter
- Visual similarity bars with percentage scores
- Real-time calculation and display

### 3. Evolutionary Algorithm
- Initialize evolution with target strings
- Configurable parameters:
  - Population size (10-200)
  - Generations per step (1-50)
  - Output length (10-500)
- Step-by-step evolution control
- Real-time fitness tracking and population display

### 4. Attention Analysis
- Analyze string components using attention mechanism
- Color-coded attention scores for each component
- Overall attention score calculation
- Visual heatmap representation

### 5. Visualization Components
- **Similarity Chart**: Bar chart visualization of cosine similarities
- **Evolution Progress**: Fitness history tracking across generations
- **Attention Heatmap**: Color-coded component attention scores

## Usage

1. **Start the server**:
   ```bash
   cd src/evolvattention
   python server.py
   ```

2. **Open the web interface**:
   Navigate to `http://localhost:8042` in your browser

3. **Workflow**:
   - Calculate barycenter with input strings
   - Use cosine similarities to compare test strings
   - Initialize evolution with target strings
   - Step through evolution generations
   - Analyze attention patterns in evolved strings
   - Visualize results using the chart tools

## Technical Details

### Frontend
- **HTML5**: Semantic structure with responsive design
- **CSS3**: Modern styling with gradients, animations, and responsive grid
- **JavaScript**: ES6+ class-based architecture with async/await

### Backend API Endpoints
- `POST /api/v1/barycenter` - Calculate barycenter
- `POST /api/v1/cosine-similarities` - Calculate similarities
- `POST /api/v1/evolution/initialize` - Initialize evolution
- `POST /api/v1/evolution/step` - Step evolution
- `GET /api/v1/evolution/status` - Get evolution status
- `POST /api/v1/attention/analyze` - Analyze attention
- `GET /health` - Health check

### Design Features
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern UI**: Gradient backgrounds, card-based layout, smooth animations
- **Interactive Elements**: Loading states, error handling, real-time updates
- **Visual Feedback**: Color-coded results, progress indicators, status messages

## Browser Compatibility
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Development
The interface is built with vanilla JavaScript, HTML, and CSS for maximum compatibility and minimal dependencies. 