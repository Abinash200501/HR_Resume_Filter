from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
if api_key is None:
    raise ValueError("GROQ_API_KEY is not set in environment variables")

os.environ['GROQ_API_KEY'] = api_key

HF_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"})


llm=ChatGroq(
    model='llama-3.1-8b-instant',
    groq_api_key=os.environ['GROQ_API_KEY']
)