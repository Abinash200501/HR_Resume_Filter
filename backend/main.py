import uvicorn
import logging
import shutil
from pathlib import Path
from typing import List
from backend.services.vector_store import (load_faiss_index, build_and_save_faiss_index)
from contextlib import asynccontextmanager
from backend.validation_schema import SearchRequest
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from backend.services.resume_search import search_resumes
from backend.groq import build_llm_prompt, llm_output

BASE_DIR = Path(__file__).resolve().parent
FAISS_STORE = BASE_DIR / "faiss_store"
FAISS_STORE.mkdir(exist_ok=True, parents=True)
RESOURCE_DIR = BASE_DIR / "resources"


VALID_JOB_ROLES = ["Machine Learning Engineer", "Data Scientist", "AI Engineer", "Service Now developer"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Server starting...")

    app.state.index, app.state.metadata = load_faiss_index()
    
    if app.state.index:
        logging.info("Index loaded from disk")
    else:
        logging.info("Index not found in local")

    yield
    logging.info("Server shutting down...")

app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: SearchRequest, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input. Please provide valid job_role and experience."}
    )



@app.get("/home")
def home():
    return "Welcome to resume filtering system"


@app.post("/upload")
def upload_files(files: List[UploadFile] = File(...)) -> List[dict]:
    results = []

    RESOURCE_DIR.mkdir(exist_ok=True, parents=True)

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append({"file_name": file.filename, "error": "Not a PDF file"})
            continue

        file_path = RESOURCE_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        results.append({"file_name": file.filename, "status": "uploaded"})

    # Build FAISS index only after saving valid files
    if any(r.get("status") == "uploaded" for r in results):
        try:
            if app.state.index is None:
                build_and_save_faiss_index()
                app.state.index, app.state.metadata = load_faiss_index()
                logging.info("FAISS index built successfully after upload.")
        except Exception as e:
            logging.error(f"Failed to build FAISS index: {e}")
            raise HTTPException(status_code=500, detail="Error building search index")
    else:
        logging.error("No valid PDFs uploaded")
        raise HTTPException(status_code=400, detail="No valid PDFs uploaded")

    return results


@app.post("/search-and-analyse")
def search_and_analyse(request: SearchRequest):
    if request.job_role not in VALID_JOB_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid job role. Valid roles: {VALID_JOB_ROLES}")

    if app.state.index is None:
        raise HTTPException(status_code=400, detail="No resumes indexed yet. Upload resumes first.")

    avg_resume_scores, results = search_resumes(
        request.job_role,
        request.experience,
        app.state.index,
        app.state.metadata,
        k=20
    )

    prompt = build_llm_prompt(request.job_role, request.experience, avg_resume_scores, results)
    llm_response = llm_output(prompt, request.job_role, request.experience)

    return {
        "llm_response": llm_response,
        "average_scores": avg_resume_scores
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000)