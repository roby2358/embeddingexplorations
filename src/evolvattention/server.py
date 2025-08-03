from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pathlib, textwrap
from pathlib import Path

# Import VecBookIndex only when needed (not at module level)
# from .vecx.vecbook_index import VecBookIndex

# Import EvolutionaryAlgorithm
# from .evolution import EvolutionaryAlgorithm

# ensure static directory exists at import-time so Starlette StaticFiles doesn't raise
_STATIC_ROOT = pathlib.Path("resources/public")
_STATIC_ROOT.mkdir(parents=True, exist_ok=True)

_INDEX = _STATIC_ROOT / "index.html"
if not _INDEX.exists():
    _INDEX.write_text(
        textwrap.dedent(
            """
            <!DOCTYPE html><html><head>
              <meta charset='utf-8'/>
              <title>EvolvAttention</title>
            </head><body>
              <h1>EvolvAttention API</h1>
              <p>The server is running.</p>
            </body></html>
            """
        ).strip()
    )

# ────────────────────────────────────────────────────────────────────────────────
# Data-models
# ────────────────────────────────────────────────────────────────────────────────

class BarycenterRequest(BaseModel):
    strings: List[str]


class CosineSimilaritiesRequest(BaseModel):
    strings: List[str]


class EvolutionInitRequest(BaseModel):
    target_strings: List[str]
    population_size: int = 50
    step_generations: int = 10
    output_length: int = 100


class AttentionAnalyzeRequest(BaseModel):
    string: str


# ────────────────────────────────────────────────────────────────────────────────
# In-memory state (single-user, non-persistent)
# ────────────────────────────────────────────────────────────────────────────────

class ServerState:  # noqa: D101 – simple state holder
    barycenter_vector: Optional[list] = None
    evolution_session: Optional[dict] = None
    vecbook_index: Optional[object] = None  # Changed from VecBookIndex to object
    evolutionary_algorithm: Optional[object] = None  # EvolutionaryAlgorithm instance


state = ServerState()

# Initialize VecBookIndex instance
def get_vecbook_index():
    """Get or create VecBookIndex instance"""
    if state.vecbook_index is None:
        # Import VecBookIndex only when needed
        from .vecx.vecbook_index import VecBookIndex
        
        # Create data directory if it doesn't exist
        data_path = Path("data")
        data_path.mkdir(exist_ok=True)
        
        # Initialize VecBookIndex
        state.vecbook_index = VecBookIndex(data_path)
    
    return state.vecbook_index


# Initialize EvolutionaryAlgorithm instance
def get_evolutionary_algorithm():
    """Get or create EvolutionaryAlgorithm instance"""
    if state.evolutionary_algorithm is None:
        # Import EvolutionaryAlgorithm only when needed
        from .evolution import EvolutionaryAlgorithm
        
        # Get VecBookIndex instance
        vecbook_index = get_vecbook_index()
        
        # Initialize EvolutionaryAlgorithm
        state.evolutionary_algorithm = EvolutionaryAlgorithm(vecbook_index)
    
    return state.evolutionary_algorithm


# ────────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ────────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="EvolvAttention API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Mount static files at root, but with check_files=False to allow API routes to take precedence
# ────────────────────────────────────────────────────────────────────────────────
# Routes – Connected to VecBookIndex methods
# ────────────────────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> FileResponse:  # noqa: D401 – FastAPI naming
    """Serve the main HTML page."""
    return FileResponse("resources/public/index.html")


@app.get("/index.css")
async def serve_css() -> FileResponse:
    """Serve the CSS file."""
    return FileResponse("resources/public/index.css")


@app.get("/index.js")
async def serve_js() -> FileResponse:
    """Serve the JavaScript file."""
    return FileResponse("resources/public/index.js")


@app.post("/api/v1/barycenter")
async def barycenter(req: BarycenterRequest):  # noqa: D401
    """Calculate cosine barycenter of multiple strings and store in server memory"""
    try:
        vecbook = get_vecbook_index()
        
        # Use VecBookIndex to set target strings and calculate barycenter
        result = vecbook.set_target_strings(req.strings)
        
        if result["status"] == "success":
            # Store barycenter info in state for other endpoints
            state.barycenter_vector = {
                "target_strings": req.strings,
                "target_count": result["target_count"],
                "barycenter_dimension": result["barycenter_dimension"]
            }
            return {"status": "success"}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating barycenter: {str(e)}")


@app.post("/api/v1/cosine-similarities")
async def cosine(req: CosineSimilaritiesRequest):  # noqa: D401
    """Calculate cosine similarities between input strings and stored barycenter"""
    if state.barycenter_vector is None:
        raise HTTPException(status_code=400, detail="No barycenter set. Call /barycenter first.")
    
    try:
        vecbook = get_vecbook_index()
        
        # Use VecBookIndex to compare strings against stored barycenter
        results = vecbook.compare_against_barycenter(req.strings)
        
        if results:
            # Extract similarities in the same order as input strings
            similarities = [float(result["cosine_similarity"]) for result in results]
            return {
                "status": "success", 
                "data": {"similarities": similarities}
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to calculate similarities")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating similarities: {str(e)}")


@app.post("/api/v1/evolution/initialize")
async def evolution_initialize(req: EvolutionInitRequest):  # noqa: D401
    """Initialize evolutionary algorithm with target strings."""
    try:
        # Get evolutionary algorithm instance
        ea = get_evolutionary_algorithm()
        
        # Update population size from request
        ea.update_population_size(req.population_size)
        
        # Initialize population using evolutionary algorithm
        init_result = ea.initialize_population(req.target_strings)
        if init_result["status"] != "success":
            raise HTTPException(status_code=400, detail=init_result["message"])
        
        # Store evolution session parameters
        state.evolution_session = {
            "target_strings": req.target_strings,
            "population_size": req.population_size,
            "step_generations": req.step_generations,
            "output_length": req.output_length,
            "generation": 0
        }
        
        # Get population data for response
        population_data = ea.get_population_data()
        
        return {
            "status": "success",
            "data": {
                "population_size": req.population_size,
                "step_generations": req.step_generations,
                "output_length": req.output_length,
                "population": population_data
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing evolution: {str(e)}")


@app.post("/api/v1/evolution/step")
async def evolution_step():  # noqa: D401
    """Run multiple generations of evolution."""
    if state.evolution_session is None:
        raise HTTPException(status_code=400, detail="No active evolution session")
    
    try:
        # Get evolutionary algorithm instance
        ea = get_evolutionary_algorithm()
        
        # Run multiple generations
        step_gens = state.evolution_session["step_generations"]
        
        for _ in range(step_gens):
            # Evolve one generation
            evolve_result = ea.evolve_generation()
            if evolve_result["status"] != "success":
                raise HTTPException(status_code=500, detail=evolve_result["message"])
        
        # Update session generation count
        state.evolution_session["generation"] = ea.generation
        
        # Get population data for response
        population_data = ea.get_population_data()
        
        # Get current statistics
        status_result = ea.get_status()
        
        return {
            "status": "success",
            "data": {
                "generation": state.evolution_session["generation"],
                "best_fitness": status_result["best_fitness"],
                "average_fitness": status_result["average_fitness"],
                "median_fitness": status_result["average_fitness"],  # Use average as median for now
                "population": population_data
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evolution step: {str(e)}")


@app.get("/api/v1/evolution/status")
async def evolution_status():  # noqa: D401
    """Get current status of evolutionary session."""
    if state.evolution_session is None:
        raise HTTPException(status_code=400, detail="No active evolution session")
    
    try:
        # Get evolutionary algorithm instance
        ea = get_evolutionary_algorithm()
        
        # Get current status
        status_result = ea.get_status()
        if status_result["status"] != "success":
            raise HTTPException(status_code=500, detail=status_result["message"])
        
        return {
            "status": "success",
            "data": {
                "current_generation": status_result["current_generation"],
                "best_fitness": status_result["best_fitness"],
                "convergence_rate": status_result["convergence_rate"],
                "best_string": status_result["best_string"],
                "is_complete": status_result["is_complete"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting evolution status: {str(e)}")


@app.post("/api/v1/attention/analyze")
async def attention_analyze(req: AttentionAnalyzeRequest):  # noqa: D401
    """Analyze string components using attention mechanism against stored barycenter."""
    if state.barycenter_vector is None:
        raise HTTPException(status_code=400, detail="No barycenter available. Call /barycenter first.")
    
    try:
        vecbook = get_vecbook_index()
        
        # Simple attention analysis using VecBookIndex
        # Split string into components (words)
        components = req.string.split()
        if not components:
            components = [req.string]
        
        # Calculate attention scores for each component using VecBookIndex
        attention_components = []
        for i, component in enumerate(components):
            # Compare component against barycenter
            similarity_results = vecbook.compare_against_barycenter([component])
            score = float(similarity_results[0]["cosine_similarity"]) if similarity_results else 0.0
            
            attention_components.append({
                "text": component,
                "score": score,
                "position": i
            })
        
        # Calculate overall score
        overall_score = sum(c["score"] for c in attention_components) / len(attention_components)
        
        return {
            "status": "success",
            "data": {
                "components": attention_components,
                "overall_score": overall_score
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing attention: {str(e)}")


@app.get("/health")
async def health():  # noqa: D401
    return {"status": "healthy"}


# ────────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: D401
    import uvicorn

    uvicorn.run(app, host="localhost", port=8042)


if __name__ == "__main__":
    main()

