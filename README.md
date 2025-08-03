# Embedding Explorations

A collection of projects exploring embedding-based analysis and evolutionary algorithms.

## Projects

### EvolvAttention

A comprehensive system for evolutionary attention analysis using embedding-based string processing and genetic algorithms.

**Location**: `src/evolvattention/`

**Quick Start**:
```bash
# Install dependencies
uv sync --extra dev

# Run the server
uv run python -m src.evolvattention.server
```

**Features**:
- FastAPI server with evolutionary algorithm endpoints
- Vector processing and embedding operations
- Text record management
- Attention mechanism analysis

See [src/evolvattention/README.md](src/evolvattention/README.md) for detailed documentation.

## Development

### Prerequisites

1. Install uv (if not already installed):
```bash
pip install uv
```

2. Install dependencies:
```bash
uv sync --extra dev
```

### Code Quality

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

## Project Structure

```
embeddingexplorations/
├── README.md                    # This file
├── pyproject.toml              # Top-level project configuration
├── src/
│   └── evolvattention/         # EvolvAttention system
│       ├── README.md           # System documentation
│       ├── server/             # FastAPI server module
│       ├── vecx/               # Vector processing module
│       └── textrec/            # Text records module
└── tests/                      # Test suite
    └── evolvattention/
        └── server/
```

## License

See LICENSE file for details.
