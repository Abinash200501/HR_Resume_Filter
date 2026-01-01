from backend.services.models import HF_embeddings
import numpy as np


def search_resumes(job_role, experience, index, metadata, k):

    if index is None: 
        return {"error": "Vector index not built yet please upload resumes first."}
    
    query = f"{job_role} with {experience} experience"
    embedded_query = HF_embeddings.embed_query(query)

    embedded_query = np.array(embedded_query, dtype="float32")

    normalised_query = embedded_query / np.linalg.norm(embedded_query)
    normalised_query = normalised_query.reshape(1,-1)

    k = min(k, len(metadata))

    distances, indices = index.search(normalised_query, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        meta = metadata[idx]
        results.append({"file_name": meta['file_name'],
                        "chunk": meta["chunk"],
                        "score": float(score)})
        
    resumes_scores = {}

    for resume in results:
        file_name = resume['file_name']
        if file_name not in resumes_scores:
            resumes_scores[file_name] = [resume['score']]
        else:
            resumes_scores[file_name].append(resume['score'])


    avg_resume_score_list = {}

    for resume_name, scores in resumes_scores.items():
        avg_score = float(np.mean(scores))
        avg_resume_score_list[resume_name] = avg_score

    return avg_resume_score_list, results       