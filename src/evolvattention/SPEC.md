# EvolvAttention System Specification

## HTTP API for UI Server

### Overview
The HTTP API MUST provide JSON/HTTP endpoints for batch string processing, embedding generation, and evolutionary algorithm operations. The API MUST support JSON request/response formats and MUST implement proper error handling.

### Base Configuration
- **Base URL**: `/api/v1`
- **Content-Type**: `application/json`
- **Error Response Format**: `{"error": "message", "status": "error"}`
- **Success Response Format**: `{"status": "success", "data": {...}}`

### Endpoints

#### 1. String Processing Endpoints

**POST /api/v1/barycenter**
- **Purpose**: Calculate cosine barycenter of multiple strings and store in server memory
- **Request Body**:
  ```json
  {
    "strings": ["string1", "string2", "string3"]
  }
  ```
- **Response**:
  ```json
  {
    "status": "success"
  }
  ```
- **Requirements**: 
  - **MUST** Store the calculated barycenter vector in server memory for later operations
- **MUST** Make the barycenter available to other endpoints (evolution, attention analysis)
- **MUST** Normalize the barycenter vector for proper cosine similarity calculations
  - **Note**: This makes the server stateful, which is acceptable for local, single-user deployment

**POST /api/v1/cosine-similarities**
- **Purpose**: Calculate cosine similarities between input strings and stored barycenter
- **Request Body**:
  ```json
  {
    "strings": ["string1", "string2", "string3"]
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "data": {
      "similarities": [0.85, 0.72, 0.91]
    }
  }
  ```
- **Requirements**:
  - **MUST** Use the barycenter stored in server memory from `/barycenter` endpoint
- **MUST** Return similarities in the same order as input strings
- **MUST** Return 400 error if no barycenter is stored

#### 2. Evolutionary Algorithm Endpoints

**POST /api/v1/evolution/initialize**
- **Purpose**: Initialize evolutionary algorithm with target strings
- **Request Body**:
  ```json
  {
    "target_strings": ["target1", "target2"],
    "population_size": 50,
    "step_generations": 10,
    "output_length": 100
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "data": {
      "population_size": 50,
      "step_generations": 10,
      "output_length": 100,
      "population": [
        {"string": "initial string 1", "similarity": 0.85},
        {"string": "initial string 2", "similarity": 0.72},
        {"string": "initial string 3", "similarity": 0.91}
      ]
    }
  }
  ```
- **Requirements**:
  - **MUST** Generate initial population from target strings with random variations
- **MUST** Store initial population in server memory with generation count (0)
- **MUST** Store target strings and barycenter in server memory
- **MUST** Calculate cosine similarities for all population members
- **MUST** Return all initial population strings with their cosine similarities to the target barycenter
- **MUST** Return population strings in order of decreasing similarity (best first)
- **MUST** Handle evolutionary algorithm logic on server side

**POST /api/v1/evolution/step**
- **Purpose**: Run multiple generations of evolution
- **Request Body**:
  ```json
  {}
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "data": {
      "generation": 10,
      "best_fitness": 0.85,
      "average_fitness": 0.72,
      "median_fitness": 0.67,
      "population": [
        {"string": "evolved string 1", "similarity": 0.85},
        {"string": "evolved string 2", "similarity": 0.72},
        {"string": "evolved string 3", "similarity": 0.91}
      ]
    }
  }
  ```
- **Requirements**:
  - **MUST** Run the number of generations specified by `step_generations` from initialization
- **MUST** Update stored population in server memory with evolved individuals
- **MUST** Update generation count in server memory
- **MUST** Calculate cosine similarities for all evolved population members
- **MUST** Return all population strings with their cosine similarities to the target barycenter
- **MUST** Return population strings in order of decreasing similarity (best first)
- **MUST** Handle evolutionary algorithm logic on server side

**GET /api/v1/evolution/status**
- **Purpose**: Get current status of evolutionary session
- **Response**:
  ```json
  {
    "status": "success",
    "data": {
      "current_generation": 15,
      "best_fitness": 0.89,
      "convergence_rate": 0.05,
      "best_string": "output N",
      "is_complete": false
    }
  }
  ```
- **Requirements**:
  - **MUST** Read current generation count from server memory
- **MUST** Calculate current best fitness from stored population
- **MUST** Calculate convergence rate from recent fitness history
- **MUST** Return 400 error if no evolution session is active

#### 3. Attention Mechanism Endpoints

**POST /api/v1/attention/analyze**
- **Purpose**: Analyze string components using attention mechanism against stored barycenter
- **Request Body**:
  ```json
  {
    "string": "string to analyze"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "data": {
      "components": [
        {"text": "component1", "score": 0.85, "position": 0},
        {"text": "component2", "score": 0.72, "position": 1}
      ],
      "overall_score": 0.78
    }
  }
  ```
- **Requirements**:
  - **MUST** Use the barycenter stored in server memory from `/barycenter` endpoint
- **MUST** Analyze each string's components against the stored barycenter
- **MUST** Return attention scores for each component
- **MUST** Return 400 error if no barycenter is stored

### Error Handling

**400 Bad Request**
- Invalid JSON in request body
- Missing required fields
- Invalid parameter values

**404 Not Found**
- Invalid session_id for evolutionary operations
- Endpoint not found

**500 Internal Server Error**
- Embedding model loading failure
- Evolutionary algorithm errors
- Database/storage errors

### CORS
- **MUST** Enable CORS for UI domain
- **SHOULD**: Allow credentials for authenticated endpoints

### Security
- **MUST** Validate all input strings (max length, character encoding)
- **SHOULD**: Implement request size limits (max 1MB per request)

## Programmatic API for VecX Classes

### Overview
The VecX classes MUST provide a clean, object-oriented interface for vector operations, embedding generation, and evolutionary algorithm support. The design MUST emphasize encapsulation, single responsibility, and extensibility.

### Core Classes

#### VecBookIndex Class

**Purpose**: Primary interface for vector indexing and similarity operations

**Constructor**:
```python
def __init__(self, path: Path, max_results: int = 10, 
             embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
             similarity_metric: str = "cosine")
```

**MUST Implement Methods**:

`build_index() -> Dict[str, Any]`
- Builds document index from all files in data directory
- Returns status dict with success/error information
- MUST handle file discovery and parsing errors gracefully

`search(query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]`
- Performs vector similarity search
- MUST use vector search when available, fallback to simple search
- MUST return standardized result format

`embed_texts(texts: List[str]) -> List[List[float]]`
- Generates embeddings for list of texts
- MUST return list of vector representations
- MUST handle embedding generation errors

`set_target_strings(target_strings: List[str]) -> Dict[str, Any]`
- Sets target strings and calculates barycenter
- MUST store target embeddings and barycenter
- MUST return status information

`cosine_barycenter_similarity(target_strings: List[str], test_strings: List[str]) -> List[Dict[str, Any]]`
- Calculates cosine barycenter of target strings
- MUST return similarities with test strings
- MUST normalize vectors for cosine similarity

`compare_against_barycenter(test_strings: List[str]) -> List[Dict[str, Any]]`
- Compares test strings against stored barycenter
- MUST use stored target barycenter
- MUST return standardized comparison results

**SHOULD Implement Methods**:

`get_target_info() -> Dict[str, Any]`
- Returns information about currently stored targets
- SHOULD include target count and barycenter dimension

`search_vector(query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]`
- Performs vector similarity search using FAISS
- SHOULD handle query embedding generation
- SHOULD normalize for cosine similarity when needed

`search_simple(query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]`
- Simple text-based search fallback
- SHOULD implement keyword-based matching
- SHOULD calculate simple relevance scores

**Private Methods**:

`_initialize_model()`
- MUST initialize sentence transformer model lazily
- MUST handle model loading errors

`_create_embeddings(texts: List[str]) -> np.ndarray`
- MUST generate embeddings in batches for efficiency
- MUST return numpy array of embeddings

`_build_faiss_index(embeddings: np.ndarray)`
- MUST build FAISS index from embeddings
- MUST handle different similarity metrics (cosine/L2)

#### TextRecords Class

**Purpose**: Handles discovery and parsing of text files with record separators

**Constructor**:
```python
def __init__(self, path: Path)
```

**MUST Implement Methods**:

`discover_files() -> List[Path]`
- Discovers all .txt files in path recursively
- MUST return empty list if path doesn't exist
- MUST handle file system errors gracefully

`parse_file(file_path: Path) -> List[Dict[str, Any]]`
- Parses single file and extracts records separated by '---'
- MUST return standardized record format
- MUST handle file reading and parsing errors

**Record Format**:
```python
{
    "text": str,                    # Record content
    "file_path": str,               # Relative path to source file
    "byte_offset": int,             # Approximate byte offset
    "record_index": int,            # Record index within file
    "metadata": {
        "source_file": str,         # Source file path
        "record_number": int,       # 1-based record number
        "total_records_in_file": int # Total records in source file
    }
}
```

### Design Principles

**Encapsulation**:
- MUST hide internal implementation details
- MUST provide clean public interfaces
- MUST encapsulate model loading and FAISS operations

**Single Responsibility**:
- VecBookIndex MUST focus on vector operations and similarity
- TextRecords MUST focus on file parsing and record extraction
- MUST separate concerns between indexing and file handling

**Extensibility**:
- MUST support different embedding models via configuration
- MUST support different similarity metrics
- SHOULD allow for custom record parsing strategies

**Error Handling**:
- MUST provide meaningful error messages
- MUST handle file system errors gracefully
- MUST handle model loading failures
- MUST return standardized error responses

**Performance**:
- MUST implement batch processing for embeddings
- MUST use efficient FAISS indexing
- SHOULD implement lazy loading for models
- SHOULD cache frequently used embeddings

**Configuration**:
- MUST support configurable embedding models
- MUST support configurable similarity metrics
- MUST support configurable result limits
- SHOULD support environment-based configuration

### Integration Points

**Evolutionary Algorithm Integration**:
- VecBookIndex MUST provide barycenter calculation for fitness evaluation
- MUST support target string management for evolution sessions
- SHOULD provide efficient comparison methods for population evaluation

**API Server Integration**:
- VecBookIndex MUST provide clean interfaces for HTTP endpoints
- MUST return JSON-serializable data structures
- MUST handle concurrent access safely

**UI Integration**:
- VecBookIndex MUST provide data suitable for visualization
- MUST support real-time search and comparison operations
- SHOULD provide progress information for long-running operations

## Evolutionary Algorithm Behavior

### Overview
The evolutionary algorithm MUST evolve text strings toward a target barycenter using attention-enhanced crossover and mutation operations. The algorithm MUST use cosine similarity as the primary fitness metric and MUST implement generational evolution with elitism.

### Core Algorithm Components

#### Population Management

**Population Structure**:
```python
class Population:
    def __init__(self, size: int, target_barycenter: np.ndarray)
    individuals: List[Individual]  # Current population
    target_barycenter: np.ndarray  # Target vector to evolve toward
    generation: int                # Current generation number
    best_fitness: float           # Best fitness achieved
    average_fitness: float        # Average fitness of population
```

**Individual Structure**:
```python
class Individual:
    def __init__(self, text: str, embedding: np.ndarray)
    text: str                     # The text string
    embedding: np.ndarray         # Vector representation
    fitness: float               # Cosine similarity to target
    attention_scores: List[float] # Component attention scores
```

**MUST Implement Operations**:

`initialize_population(target_strings: List[str], population_size: int) -> Population`
- MUST create initial population from target strings
- MUST generate random variations for diversity
- MUST calculate initial fitness scores
- MUST set target barycenter from input strings

`evaluate_fitness(individual: Individual, target_barycenter: np.ndarray) -> float`
- MUST calculate cosine similarity between individual and target
- MUST normalize vectors before comparison
- MUST return value between 0.0 and 1.0
- MUST handle embedding generation errors

`select_parents(population: Population, tournament_size: int = 3) -> Tuple[Individual, Individual]`
- MUST implement tournament selection
- MUST favor higher fitness individuals
- MUST return two distinct parents
- SHOULD use configurable tournament size

#### Crossover Operations

**Attention-Enhanced Crossover**:
```python
def attention_crossover(parent1: Individual, parent2: Individual, 
                       attention_model: AttentionModel) -> Individual
```

**Note**: In the first implementation pass, this SHOULD be a simple crossover function that combines parent strings with 50/50 chance for each character. Attention mechanism integration is a future enhancement goal.

**(future) MUST Implement Behavior**
- MUST analyze parent strings using attention mechanism
- MUST identify high-value components from each parent
- MUST preserve valuable components in offspring
- MUST combine components intelligently
- MUST generate new embedding for offspring
- MUST handle crossover failures gracefully

**Component Analysis**:
- MUST break strings into meaningful components (words, phrases)
- MUST calculate attention scores for each component
- MUST identify components with high similarity to target
- SHOULD consider component position and context

**Crossover Strategies**:
- MUST implement weighted component selection
- MUST preserve semantic coherence
- MUST avoid duplicate components when possible
- SHOULD implement multiple crossover strategies

#### Mutation Operations

**String-Based Mutations**:
```python
def mutate_individual(individual: Individual, mutation_rate: float) -> Individual
```

**MUST Implement Mutations**:
- **Character-level**: Random character substitution
- (future) **Word-level**: Word insertion, deletion, replacement
- (future) **Phrase-level**: Phrase reordering, replacement
- (future) **Semantic-level**: Synonym replacement, paraphrasing

**Mutation Control**:
- MUST use configurable mutation rates
- MUST maintain string validity
- MUST avoid excessive mutations
- SHOULD implement adaptive mutation rates

#### Attention Mechanism Integration

**Attention Model**:
```python
class AttentionModel:
    def __init__(self, target_barycenter: np.ndarray)
    target_barycenter: np.ndarray  # Target vector
    component_analyzer: ComponentAnalyzer  # Component analysis
```

**Component Analysis**:
```python
def analyze_components(text: str, target_barycenter: np.ndarray) -> List[Component]
    # Returns list of components with attention scores
```

**MUST Implement Features**:
- MUST identify meaningful text components
- MUST calculate component relevance to target
- MUST provide attention scores for crossover
- MUST support multiple component granularities
- SHOULD implement context-aware analysis

### Algorithm Flow

#### Initialization Phase
1. **MUST** Accept target strings and calculate barycenter
2. **MUST** Initialize population with target variations
3. **MUST** Calculate initial fitness for all individuals
4. **MUST** Set up attention model with target barycenter

#### Evolution Loop
1. **MUST** Evaluate fitness for all individuals
2. **MUST** Select parents using tournament selection
3. **MUST** Perform attention-enhanced crossover
4. **MUST** Apply mutation operations
5. **MUST** Evaluate offspring fitness
6. **MUST** Replace population using elitism
7. **MUST** Update generation statistics

#### Termination Conditions
- **MUST** Support maximum generation limit
- **MUST** Support fitness convergence threshold
- **MUST** Support minimum fitness improvement
- **SHOULD**: Support early convergence detection

### Configuration Parameters

**Population Parameters**:
- `population_size`: Number of individuals (MUST be >= 10)
- `elite_size`: Number of best individuals to preserve (MUST be >= 1)
- `tournament_size`: Tournament selection size (MUST be >= 2)

**Evolution Parameters**:
- `crossover_rate`: Probability of crossover (MUST be 0.0-1.0)
- `mutation_rate`: Probability of mutation (MUST be 0.0-1.0)
- `max_generations`: Maximum generations to run (MUST be > 0)

**Fitness Parameters**:
- `convergence_threshold`: Minimum fitness improvement (MUST be > 0.0)
- `target_fitness`: Target fitness to achieve (MUST be 0.0-1.0)

**Attention Parameters**:
- `attention_threshold`: Minimum attention score for preservation (MUST be 0.0-1.0)
- `component_granularity`: Component analysis level (MUST be "word", "phrase", "sentence")

### Performance Requirements

**Scalability**:
- MUST handle populations of 100+ individuals
- MUST support 1000+ generations
- MUST complete generation in < 10 seconds
- SHOULD implement parallel fitness evaluation

**Memory Management**:
- MUST avoid memory leaks during evolution
- MUST limit embedding cache size
- SHOULD implement garbage collection for old individuals

**Convergence**:
- MUST detect when fitness stops improving
- MUST provide convergence statistics
- SHOULD implement restart mechanisms for local optima

### Integration with VecBookIndex

**Fitness Evaluation**:
- MUST use VecBookIndex for embedding generation
- MUST use VecBookIndex for barycenter calculations
- MUST handle VecBookIndex errors gracefully

**Session Management**:
- MUST maintain evolution session state
- MUST provide session persistence
- MUST support session resumption

**Progress Reporting**:
- MUST provide real-time progress updates
- MUST report generation statistics
- MUST provide best individual information
- SHOULD provide population diversity metrics
