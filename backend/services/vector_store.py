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
    
def load_existing_index(embedding_dim):
    if index_path.exists() and metadata_path.exists():
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata
    logging.info("Content got changed so building index")
    index = faiss.IndexFlatIP(embedding_dim)
    return index, []

def build_and_save_faiss_index() -> bool:
    embedded_chunk = embedding_chunk()

    if not embedded_chunk:
        logging.info("No resumes found. Skipping FAISS index build.")
        return False

    if not embedded_chunk[0].get("embeddings"):
        logging.info("No embeddings generated. Skipping FAISS index build.")
        return False

    embedding_dim = len(embedded_chunk[0]["embeddings"][0]["embedding"])

    index, metadata = load_existing_index(embedding_dim)
    indexed_doc_ids = {m["doc_id"] for m in metadata}
    next_faiss_id = len(metadata)

    new_vectors = []
    new_metadata = []

    for file_data in embedded_chunk:
        try:
            doc_id = file_data["doc_id"]
        except KeyError:
            logging.error(f"Missing doc_id in file_data: {file_data}")
            continue

        if doc_id in indexed_doc_ids:
            logging.info(f"Skipping already indexed document: {file_data['file_name']}")
            continue

        for chunk in file_data["embeddings"]:
            vec = np.array(chunk["embedding"], dtype="float32")
            norm = np.linalg.norm(vec)

            if norm > 0:
                vec = vec / norm

            new_vectors.append(vec)

            new_metadata.append({
                "faiss_index": next_faiss_id,
                "file_name": file_data["file_name"],
                "doc_id": file_data['doc_id'],
                "index": chunk["index"],
                "chunk": chunk["chunk"],
            })

            next_faiss_id += 1

    if not new_vectors:
        logging.info("No new documents to index.")
        return False

    index.add(np.vstack(new_vectors))
    metadata.extend(new_metadata)

    faiss.write_index(index, str(index_path))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("FAISS index built successfully")
    return True

def load_faiss_index():

    if not index_path.exists() or not metadata_path.exists():
        return None, []

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata