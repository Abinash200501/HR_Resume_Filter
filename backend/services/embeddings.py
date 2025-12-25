from backend.services.models import HF_embeddings
from backend.services.chunker import chunk_resumes


resume_chunks = chunk_resumes() # list of dictionary and chunks_of_resume is list of nested dictionary

def embedding_chunk():
    embedding_metadata = []
    for chunk in resume_chunks:
        texts = [text['chunk'] for text in chunk['chunks_of_resume']]  # iterating over a list of dictionary
        embeddings = HF_embeddings.embed_documents(texts=texts)

        # for each chunk each embedding got mapped with its file name
        embedding_metadata.append({"file_name": chunk['file_name'], 
                                   "embeddings":[{"index":text['index'], "chunk": text['chunk'], "embedding": emb}
                                                for text, emb in zip(chunk['chunks_of_resume'], embeddings)]})
    return embedding_metadata
    