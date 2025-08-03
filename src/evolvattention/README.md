# EvolvAttention System

A comprehensive system for evolutionary attention analysis using embedding-based string processing and genetic algorithms.

## Project Overview

EvolvAttention is designed to analyze and evolve text strings using attention mechanisms and evolutionary algorithms. The system consists of multiple components:

- **FastAPI Server** (`server/`) - HTTP API for batch string processing and evolution
- **Vector Processing** (`vecx/`) - Embedding and vector operations
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
uv run uvicorn src.evolvattention.server.app.server:app --host 0.0.0.0 --port 8000

# Option 3: Run from server directory
cd src/evolvattention/server
uv run python -m server
```

The server will start on `http://localhost:8000`

## System Architecture

### FastAPI Server (`server/`)

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

### Vector Processing (`vecx/`)

Core vector operations and embedding functionality:

- Embedding generation and storage
- Cosine similarity calculations
- Vector normalization and barycenter computation
- Evolutionary algorithm vector operations

### Text Records (`textrec/`)

Text processing and record management:

- Text record structures and validation
- String preprocessing and tokenization
- Record persistence and retrieval

## Development

### Project Structure

```
src/evolvattention/
├── README.md              # This file
├── SPEC.md               # Technical specifications
├── server/               # FastAPI server module
│   ├── __init__.py       # Module initialization
│   ├── __main__.py       # Entry point
│   ├── pyproject.toml    # Project configuration
│   ├── app/              # Application package
│   │   ├── __init__.py   # App package init
│   │   └── server.py     # Main server code
│   └── tests/            # Test suite
├── vecx/                 # Vector processing module
│   ├── __init__.py
│   ├── vecbook_index.py  # Vector indexing
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

This is a **stub implementation** with the following characteristics:

- ✅ All API endpoints implemented according to SPEC.md
- ✅ Proper error handling and validation
- ✅ CORS middleware configured
- ✅ Static file serving
- ✅ In-memory state management for single-user deployment
- ✅ Modern Python packaging with uv and pyproject.toml
- ✅ Comprehensive test suite
- 🔄 **Stub implementations** - Real embedding generation and evolutionary algorithms need to be implemented

## Testing

You can test the API using curl or any HTTP client:

```bash
# Test barycenter calculation
curl -X POST "http://localhost:8000/api/v1/barycenter" \
  -H "Content-Type: application/json" \
  -d '{"strings": ["hello world", "goodbye world", "test string"]}'

# Test evolution initialization
curl -X POST "http://localhost:8000/api/v1/evolution/initialize" \
  -H "Content-Type: application/json" \
  -d '{"target_strings": ["target1", "target2"], "population_size": 10}'

# Test attention analysis
curl -X POST "http://localhost:8000/api/v1/attention/analyze" \
  -H "Content-Type: application/json" \
  -d '{"string": "test string to analyze"}'
```

## Next Steps

1. **Implement real embedding generation** using a language model (e.g., sentence-transformers)
2. **Implement actual evolutionary algorithm** with crossover and mutation operators
3. **Implement attention mechanism** for string component analysis
4. **Add proper vector storage** and similarity calculations
5. **Integrate vecx and textrec modules** with the server
6. **Add comprehensive testing** for all components
7. **Implement web UI** for visualization and interaction

## Contributing

1. Follow the existing code structure and patterns
2. Add tests for new functionality
3. Use the development tools (black, isort, mypy) to maintain code quality
4. Update documentation as needed

## License

See LICENSE file for details.
