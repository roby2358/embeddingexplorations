# EvolvAttention System

A comprehensive system for evolutionary attention analysis using embedding-based string processing and genetic algorithms.

## Project Overview

EvolvAttention is designed to analyze and evolve text strings using attention mechanisms and evolutionary algorithms. The system consists of multiple components:

- **FastAPI Server** (`server.py`) - HTTP API for batch string processing and evolution
- **Evolutionary Algorithm** (`evolve/`) - Population management and genetic operations
- **Vector Processing** (`vecx/`) - Embedding and vector operations using sentence-transformers
- **Text Records** (`textrec/`) - Text processing and record management

## Quick Start

### Prerequisites

1. Install uv (if not already installed):
```bash
pip install uv
```

2. Install dependencies from the root directory:
```bash
uv sync --extra dev
```

3. Run the server:
```bash
# Option 1: Run as module from root
uv run python -m src.evolvattention.server

# Option 2: Run with uvicorn directly
uv run uvicorn src.evolvattention.server:app --host 0.0.0.0 --port 8042

# Option 3: Run from server directory
cd src/evolvattention
uv run python server.py
```

The server will start on `http://localhost:8042`

## System Architecture

### FastAPI Server (`server.py`)

The main HTTP API providing endpoints for:

- **String Processing**: Calculate barycenter and cosine similarities
- **Evolutionary Algorithm**: Initialize, step, and monitor evolution sessions  
- **Attention Mechanism**: Analyze string components with attention scores
- **Static File Serving**: Serves HTML/CSS/JS files from `resources/public/`

#### API Endpoints

**Base URL**: `/api/v1`

**String Processing**:
- `POST /barycenter` - Calculate and store barycenter of multiple strings
- `POST /cosine-similarities` - Calculate similarities between strings and stored barycenter

**Evolutionary Algorithm**:
- `POST /evolution/initialize` - Initialize evolution session with target strings
- `POST /evolution/step` - Run multiple generations of evolution
- `GET /evolution/status` - Get current evolution session status

**Attention Mechanism**:
- `POST /attention/analyze` - Analyze string components with attention scores

**Other**:
- `GET /` - Serve the main HTML page
- `GET /health` - Health check endpoint

### Evolutionary Algorithm (`evolve/`)

Core evolutionary algorithm implementation with:

- **Population Management**: Individual and Population classes with fitness tracking
- **Genetic Operations**: Crossover and mutation operators
- **Fitness Evaluation**: Cosine similarity to target barycenter
- **Selection**: Tournament selection and elitism
- **Generation Evolution**: Multi-generation evolution with statistics

**Key Features**:
- Population size: 50 (configurable)
- Elite size: 5 (best individuals preserved)
- Tournament size: 3 (selection pressure)
- Crossover rate: 0.8
- Mutation rate: 0.1
- Character-level mutations (substitution, insertion, deletion)

### Vector Processing (`vecx/`)

Core vector operations and embedding functionality:

- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` model for embeddings
- **FAISS Integration**: Vector similarity search and indexing
- **Barycenter Calculation**: Cosine barycenter computation with normalization
- **Similarity Metrics**: Cosine similarity calculations
- **Target Management**: Store and compare against target strings

### Text Records (`textrec/`)

Text processing and record management:

- **File Discovery**: Recursive .txt file discovery
- **Record Parsing**: Parse files with `---` separators
- **Metadata Tracking**: File paths, byte offsets, record indices
- **Error Handling**: Robust file parsing with logging

## Development

### Project Structure

```
src/evolvattention/
├── README.md              # This file
├── SPEC.md               # Technical specifications
├── server.py             # FastAPI server implementation
├── evolve/               # Evolutionary algorithm module
│   ├── __init__.py       # Module initialization
│   └── evolution.py      # Core evolutionary algorithm
├── vecx/                 # Vector processing module
│   ├── __init__.py
│   ├── vecbook_index.py  # Vector indexing and embedding
│   └── README.md         # Vector module docs
└── textrec/              # Text records module
    ├── __init__.py
    └── text_records.py   # Text record structures
```

### Code Quality

From the root directory:
```bash
# Format code
uv run black .

# Sort imports
uv run isort .

# Type checking
uv run mypy .

# Run tests
uv run pytest
```

## Current Implementation Status

### ✅ **Fully Implemented**

- **FastAPI Server**: Complete HTTP API with all endpoints
- **Error Handling**: Comprehensive error handling and validation
- **CORS Middleware**: Configured for web UI integration
- **Static File Serving**: HTML/CSS/JS file serving
- **State Management**: In-memory state for single-user deployment
- **Modern Python Packaging**: uv and pyproject.toml configuration
- **Test Suite**: Comprehensive test coverage
- **Vector Processing**: Real embedding generation using sentence-transformers
- **FAISS Integration**: Vector similarity search and indexing
- **Barycenter Calculation**: Cosine barycenter with proper normalization
- **Evolutionary Algorithm**: Complete genetic algorithm implementation
- **Population Management**: Individual and Population classes
- **Genetic Operations**: Crossover and mutation operators
- **Fitness Evaluation**: Cosine similarity to target barycenter
- **Text Records**: File discovery and record parsing

### 🔄 **Partially Implemented**

- **Attention Mechanism**: Basic implementation using component similarity
  - Currently splits strings into words and compares each against barycenter
  - Needs enhancement with proper attention scoring algorithms
  - Task: `fa-hdcb-t5` - Implement attention mechanism for analyzing string components

### ❌ **Not Yet Implemented**

- **Attention-Enhanced Crossover**: Basic crossover exists, needs attention integration
  - Task: `yd-q73q-xq` - Create attention-enhanced crossover function
- **Comprehensive Logging**: Basic logging exists, needs system-wide implementation
  - Task: `8f-uaf5-wn` - Implement proper logging throughout the system
- **Unit Tests**: Core functionality needs testing
  - Task: `65-wva7-we` - Add unit tests for VecBookIndex methods
- **Integration Tests**: API endpoint testing needed
  - Task: `75-gwpd-dk` - Create integration tests for all API endpoints
- **Documentation**: API documentation needed
  - Task: `wz-upvj-fr` - Write comprehensive documentation

## Testing

You can test the API using curl or any HTTP client:

```bash
# Test barycenter calculation
curl -X POST "http://localhost:8042/api/v1/barycenter" \
  -H "Content-Type: application/json" \
  -d '{"strings": ["hello world", "goodbye world", "test string"]}'

# Test evolution initialization
curl -X POST "http://localhost:8042/api/v1/evolution/initialize" \
  -H "Content-Type: application/json" \
  -d '{"target_strings": ["target1", "target2"], "population_size": 10}'

# Test attention analysis
curl -X POST "http://localhost:8042/api/v1/attention/analyze" \
  -H "Content-Type: application/json" \
  -d '{"string": "test string to analyze"}'
```

## Key Features

### Real Embedding Generation
- Uses `sentence-transformers/all-MiniLM-L6-v2` model
- Generates 384-dimensional embeddings
- Proper normalization for cosine similarity

### Working Evolutionary Algorithm
- Population initialization with random strings
- Tournament selection with elitism
- Character-level crossover and mutation
- Fitness evaluation using cosine similarity
- Multi-generation evolution with statistics

### Vector Operations
- FAISS integration for efficient similarity search
- Cosine barycenter calculation with normalization
- Target string management and comparison
- Embedding generation and storage

### Text Processing
- Recursive file discovery for .txt files
- Record parsing with `---` separators
- Metadata tracking and error handling
- Robust file handling with logging

## Next Steps

### High Priority Tasks
1. **Implement attention-enhanced crossover** (`yd-q73q-xq`)
2. **Enhance attention mechanism** (`fa-hdcb-t5`)
3. **Add comprehensive unit tests** (`65-wva7-we`)

### Medium Priority Tasks
1. **Implement proper logging** (`8f-uaf5-wn`)
2. **Write API documentation** (`wz-upvj-fr`)
3. **Create integration tests** (`75-gwpd-dk`)

### Future Enhancements
1. **Web UI Development**: Interactive visualization
2. **Multiple Embedding Models**: Support for different models
3. **Advanced Attention Mechanisms**: Transformer-based attention
4. **Performance Optimization**: Batch processing and caching
5. **Persistence**: Database storage for sessions and results

## Contributing

1. Follow the existing code structure and patterns
2. Add tests for new functionality
3. Use the development tools (black, isort, mypy) to maintain code quality
4. Update documentation as needed
5. Check the task list for current priorities

## License

See LICENSE file for details.
