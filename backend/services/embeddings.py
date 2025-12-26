from backend.services.models import HF_embeddings
from backend.services.chunker import chunk_resumes
import logging
from fastapi import HTTPException



def embedding_chunk():
    resume_chunks = chunk_resumes()  # list of dictionary and chunks_of_resume is list of nested dictionary

    embedding_metadata = []
    if resume_chunks:
        for chunk in resume_chunks:
            texts = [text['chunk'] for text in chunk['chunks_of_resume']]  # iterating over a list of dictionary
            embeddings = HF_embeddings.embed_documents(texts=texts)

            # for each chunk each embedding got mapped with its file name
            embedding_metadata.append({"file_name": chunk['file_name'], 
                                    "embeddings":[{"index":text['index'], "chunk": text['chunk'], "embedding": emb}
                                                    for text, emb in zip(chunk['chunks_of_resume'], embeddings)]})
    else:
        logging.info("No embedding chunks found for resumes")
        raise HTTPException(status_code=500)
    return embedding_metadata
    