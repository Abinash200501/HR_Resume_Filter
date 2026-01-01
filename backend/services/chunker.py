from backend.utils import read_files
import hashlib

def compute__doc_id(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fixed_chunking(text):
    chunk_size = 200
    overlap = 50
    step = chunk_size - overlap
    chunks = []
    for index, start in enumerate(range(0, len(text), step)):
        chunk = text[start : start + chunk_size]
        chunks.append({"index": index,
                        "chunk": chunk})
    
    return chunks


def chunk_resumes():
    resumes = read_files()

    all_chunks = []
    for resume in resumes:
        doc_id = compute__doc_id(resume['text'])
        chunks = fixed_chunking(resume['text'])
        all_chunks.append({
                           "file_name": resume['file_name'],
                           "chunks_of_resume": chunks,
                           "doc_id": doc_id,
                           "total_chunks": len(chunks),
                           "resume_id":resume['resume_id'],
                           })
        
    return all_chunks



