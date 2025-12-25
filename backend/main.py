import uvicorn
import logging
import shutil
from pathlib import Path
from typing import List
from backend.services.vector_store import load_faiss_index
from contextlib import asynccontextmanager
from backend.services.vector_store import build_and_save_faiss_index
from backend.validation_schema import SearchRequest
from backend.utils import read_files
from fastapi import FastAPI, File, UploadFile
from backend.services.resume_search import search_resumes
from backend.groq import build_llm_prompt, llm_output

BASE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = BASE_DIR / "resources"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Server starting...")

    resumes = read_files()

    if resumes:
        logging.info("Resumes are found....loading vector store")
        build_and_save_faiss_index()
        app.state.index, app.state.metadata = load_faiss_index()
        logging.info("FAISS index loaded")
        
    else:
        logging.error("No resumes found. Skipping index build.")
        app.state.index = None
        app.state.metadata = []

    yield
    logging.info("Server shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/home")
def home():
    return "Welcome to resume filtering system"

@app.post("/upload")
def upload_files(files : list[UploadFile] = File(...)) -> List[dict]:

    results = []
    for file in files:

        if not file.filename.lower().endswith(".pdf"):
            results.append({"file_name": file.filename, "error": "Not a PDF file"})
            continue

        RESOURCE_DIR.mkdir(exist_ok=True, parents=True)

        file_path = RESOURCE_DIR / file.filename

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        results.append({"file_name": file.filename, "status": "uploaded"})

    # rebuild index after upload
    build_and_save_faiss_index()
    app.state.index, app.state.metadata = load_faiss_index()

    return results
     
@app.post("/search-and-analyse")
def search_and_analyse(request : SearchRequest):

    if app.state.index is None:
        return {"error": "Vector index is not built yet"}

    avg_resume_scores, results = search_resumes(request.job_role, request.experience,
                                                app.state.index, app.state.metadata, k=20)
    
    prompt = build_llm_prompt(
        request.job_role,
        request.experience,
        avg_resume_scores,
        results
    )
    
    llm_response = llm_output(prompt, request.job_role,
        request.experience,)

    return {
        "llm_response": llm_response,
        "average_scores": avg_resume_scores
    }


if __name__ == "__main__":
    uvicorn.run(app=app)