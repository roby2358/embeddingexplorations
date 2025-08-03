from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pathlib, textwrap

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


state = ServerState()


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
# Routes – stub implementations
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
async def barycenter(_: BarycenterRequest):  # noqa: D401
    state.barycenter_vector = [0.1, 0.2, 0.3]  # stub
    return {"status": "success"}


@app.post("/api/v1/cosine-similarities")
async def cosine(req: CosineSimilaritiesRequest):  # noqa: D401
    if state.barycenter_vector is None:
        raise HTTPException(status_code=400, detail="No barycenter")
    sims = [0.8 for _ in req.strings]  # stub
    return {"status": "success", "data": {"similarities": sims}}


@app.post("/api/v1/evolution/initialize")
async def evolution_initialize(req: EvolutionInitRequest):  # noqa: D401
    """Initialize evolutionary algorithm with target strings."""
    # Store target strings and parameters
    state.evolution_session = {
        "target_strings": req.target_strings,
        "population_size": req.population_size,
        "step_generations": req.step_generations,
        "output_length": req.output_length,
        "generation": 0,
        "population": []
    }
    
    # Generate initial population (stub implementation)
    import random
    import string
    
    population = []
    for i in range(req.population_size):
        # Generate random string based on target strings
        base_string = random.choice(req.target_strings)
        # Add some random variation
        variation = ''.join(random.choices(string.ascii_letters + ' ', k=random.randint(5, 20)))
        evolved_string = base_string + " " + variation
        population.append({
            "string": evolved_string,
            "similarity": random.uniform(0.3, 0.9)  # Random similarity score
        })
    
    # Sort by similarity (best first)
    population.sort(key=lambda x: x["similarity"], reverse=True)
    state.evolution_session["population"] = population
    
    return {
        "status": "success",
        "data": {
            "population_size": req.population_size,
            "step_generations": req.step_generations,
            "output_length": req.output_length,
            "population": population
        }
    }


@app.post("/api/v1/evolution/step")
async def evolution_step():  # noqa: D401
    """Run multiple generations of evolution."""
    if state.evolution_session is None:
        raise HTTPException(status_code=400, detail="No active evolution session")
    
    # Simulate evolution (stub implementation)
    import random
    
    current_gen = state.evolution_session["generation"]
    step_gens = state.evolution_session["step_generations"]
    
    # Evolve population
    population = state.evolution_session["population"]
    for _ in range(step_gens):
        # Simple evolution: randomly improve some individuals
        for i in range(len(population)):
            if random.random() < 0.3:  # 30% chance of improvement
                population[i]["similarity"] = min(1.0, population[i]["similarity"] + random.uniform(0.01, 0.05))
                # Add some variation to the string
                population[i]["string"] += " " + ''.join(random.choices("abcdefghijklmnopqrstuvwxyz ", k=random.randint(1, 5)))
    
    # Sort by similarity (best first)
    population.sort(key=lambda x: x["similarity"], reverse=True)
    state.evolution_session["population"] = population
    state.evolution_session["generation"] = current_gen + step_gens
    
    # Calculate statistics
    similarities = [p["similarity"] for p in population]
    best_fitness = max(similarities)
    average_fitness = sum(similarities) / len(similarities)
    median_fitness = sorted(similarities)[len(similarities) // 2]
    
    return {
        "status": "success",
        "data": {
            "generation": state.evolution_session["generation"],
            "best_fitness": best_fitness,
            "average_fitness": average_fitness,
            "median_fitness": median_fitness,
            "population": population
        }
    }


@app.get("/api/v1/evolution/status")
async def evolution_status():  # noqa: D401
    """Get current status of evolutionary session."""
    if state.evolution_session is None:
        raise HTTPException(status_code=400, detail="No active evolution session")
    
    population = state.evolution_session["population"]
    similarities = [p["similarity"] for p in population]
    best_fitness = max(similarities)
    best_string = population[0]["string"]
    
    return {
        "status": "success",
        "data": {
            "current_generation": state.evolution_session["generation"],
            "best_fitness": best_fitness,
            "convergence_rate": 0.05,  # Stub value
            "best_string": best_string,
            "is_complete": False
        }
    }


@app.post("/api/v1/attention/analyze")
async def attention_analyze(req: AttentionAnalyzeRequest):  # noqa: D401
    """Analyze string components using attention mechanism."""
    if state.barycenter_vector is None:
        raise HTTPException(status_code=400, detail="No barycenter available")
    
    # Simple attention analysis (stub implementation)
    import random
    
    # Split string into components (words)
    components = req.string.split()
    if not components:
        components = [req.string]
    
    # Generate attention scores for each component
    attention_components = []
    for i, component in enumerate(components):
        score = random.uniform(0.1, 0.9)  # Random attention score
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

