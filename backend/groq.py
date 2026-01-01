from backend.services.models import llm
from langchain_core.prompts import PromptTemplate
import time
import logging
import mlflow

logging.basicConfig(filename="llm_usage.log", level=logging.INFO, format='%(asctime)s - %(message)s')
mlflow.set_experiment("resume_filtering")


def build_llm_prompt(job_role, experience, avg_resumes_scores, results):

    template = f"""
        You are an AI HR assistant.

        Job Role: {job_role}
        Experience Required: {experience}

        Below are resume matches with similarity scores.
        Explain briefly why each candidate is suitable.
        Summarize each resume in a few bullet points focusing on skills, experience, and achievements relevant to {job_role}.
        You can re-rank the resumes even based on the {job_role} and {experience} by not checking the similarity scores and this 
        should happen only if you are not sure or if you find contradicting resumes appears first.

        """
    for file_name, scores in avg_resumes_scores.items():
        template += f"\n Resume : {file_name}, average_similarity_score : {scores:.4f} \n"

        for resume in results:
            if resume['file_name'] == file_name:
                template += f"Content : {resume['chunk']}"

    template += "\n Return the final prompt with reasoning and the ranked list"
    prompt = PromptTemplate.from_template(template=template)

    return prompt

def llm_output(prompt, job_role, experience):
    prompt_text = prompt.format() 
    messages = [prompt_text]

    start_time = time.time()
    llm_response = llm.invoke(messages)
    end_time = time.time()
    response_time = end_time - start_time


    llm_metadata = llm_response.response_metadata

    token_usage = llm_metadata.get("token_usage", {})
    model_name = llm_metadata.get("model_name")

    with mlflow.start_run(run_name=model_name):
        #log parameters
        mlflow.log_metric("total tokens", token_usage.get('total_tokens',0))
        mlflow.log_metric("latency_sec", response_time)
        mlflow.log_metric('input tokens', token_usage.get('prompt_tokens', 0))
        mlflow.log_metric('completion_tokens', token_usage.get('completion_tokens', 0))
        mlflow.log_metric('model_time', token_usage.get('total_time'))

        mlflow.log_param("model", model_name)
        mlflow.log_param("job_role", job_role)
        mlflow.log_param("experience", experience)

        logging.info(f"Prompt Usage : {len(prompt_text)} characters")
        logging.info(f"Token usage : {token_usage.get('total_tokens'), 0}")
        logging.info(f"Response time : {response_time:.3f} seconds")

    return llm_response





    
