import faiss
from backend.services.embeddings import embedding_chunk
import numpy as np
from pathlib import Path
import json
import logging


embedded_chunk = embedding_chunk()

if embedded_chunk and embedded_chunk[0]['embeddings']:
    embedding_dim = len(embedded_chunk[0]['embeddings'][0]['embedding'])
else:
    embedding_dim = 0  # or some default value
    logging.warning("Warning: embedded_chunk is empty or missing embeddings")

# Directory creation
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_STORE = BASE_DIR / "faiss_store"
FAISS_STORE.mkdir(exist_ok=True, parents=True)
index_path = FAISS_STORE / "faiss.index"
metadata_path = FAISS_STORE / "faiss.json"

def build_and_save_faiss_index():

    if not embedded_chunk:
        logging.error("Warning: No embedded chunks found. Skipping FAISS index build.")
        return

    all_vectors = []
    metadata = []
    faiss_id = 0

    for file_data in embedded_chunk:
        for chunk in file_data['embeddings']:

            vec = np.array(chunk['embedding'], dtype="float32")
            norm = np.linalg.norm(vec)

            if not np.isclose(norm, 1.0, atol=1e-4):
                normalised_vector = vec / norm
                chunk['embedding'] = normalised_vector.tolist()
                all_vectors.append(normalised_vector)
            else:
                all_vectors.append(vec)
            
            metadata.append({"faiss_index": faiss_id,
                             "file_name": file_data['file_name'],
                             "index": chunk['index'],
                             "chunk": chunk['chunk']})
            
            faiss_id += 1

    index = faiss.IndexFlatIP(embedding_dim)
    vectors = np.vstack(all_vectors)          # for continuous memory
    index.add(vectors)
    faiss.write_index(index, str(index_path)) # FAISS don't accept Path object

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
def load_faiss_index():

    index = faiss.read_index(str(index_path))  

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

