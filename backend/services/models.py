from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

HF_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"})


llm=ChatGroq(
    model='llama-3.3-70b-versatile',
    groq_api_key=os.environ['GROQ_API_KEY']
)