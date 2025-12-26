import faiss
from backend.services.embeddings import embedding_chunk
import numpy as np
from pathlib import Path
import json
import logging


# Directory creation
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_STORE = BASE_DIR / "faiss_store"
FAISS_STORE.mkdir(exist_ok=True, parents=True)
index_path = FAISS_STORE / "faiss.index"
metadata_path = FAISS_STORE / "faiss.json"
    
def build_and_save_faiss_index() -> bool:
    embedded_chunk = embedding_chunk()

    if not embedded_chunk:
        logging.info("No resumes found. Skipping FAISS index build.")
        return False

    if not embedded_chunk[0].get("embeddings"):
        logging.info("No embeddings generated. Skipping FAISS index build.")
        return False

    embedding_dim = len(embedded_chunk[0]["embeddings"][0]["embedding"])

    all_vectors = []
    metadata = []
    faiss_id = 0

    for file_data in embedded_chunk:
        for chunk in file_data["embeddings"]:
            vec = np.array(chunk["embedding"], dtype="float32")
            norm = np.linalg.norm(vec)

            if norm > 0:
                vec = vec / norm

            all_vectors.append(vec)

            metadata.append({
                "faiss_index": faiss_id,
                "file_name": file_data["file_name"],
                "index": chunk["index"],
                "chunk": chunk["chunk"],
            })

            faiss_id += 1

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(np.vstack(all_vectors))

    faiss.write_index(index, str(index_path))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("FAISS index built successfully")
    return True

def load_faiss_index():
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("FAISS index or metadata not found")

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata