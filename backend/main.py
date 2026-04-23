from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import uvicorn
from backend.recommender import RecommenderEngine

app = FastAPI(title="Recomendador de Películas RAG")

# Resolve directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Engine Initialization
print("Initializing Recommender Engine...")
engine = RecommenderEngine(data_dir=DATA_DIR)
# In a real production setup, we might load lazily or asynchronously.
# For smooth user experience immediately, we can load it on startup:
@app.on_event("startup")
def startup_event():
    try:
        engine.load()
    except Exception as e:
        print(f"Advertencia al cargar modelo en startup: {e}")

class RecommendRequest(BaseModel):
    prompt: str
    k: int = 3

@app.post("/api/recommend")
async def get_recommendation(req: RecommendRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")
    
    try:
        result = engine.process_request(req.prompt, req.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Montar frontend
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
