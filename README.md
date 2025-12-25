# HR Resume Filtering and Analysis

This project is an AI-powered tool that helps HR teams efficiently filter, rank, and summarize resumes for a given job role. It uses NLP, machine learning, and LLMs to match resumes with job descriptions and provide reasoning for candidate selection.

---

## Features

- Analyze resumes against a specified job role.
- Calculate similarity scores for ranking candidates.
- Summarize each resume highlighting skills, experience, and achievements.
- Provide a ranked list of candidates with explanations.
- Tracks LLM usage, tokens, and response times (for analytics/logging).

---

## Tech Stack

- **Backend:** Python, FastAPI
- **Frontend:** Streamlit
- **AI/ML:** LangChain, Groq LLMs, Transformers
- **Environment Management:** `virtualenv`
- **Version Control:** Git, GitHub
- **Vector Database:** FAISS

---

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Abinash200501/HR-Resume_Filtering.git
    cd HR-Resume_Filtering

2. Create and activate a virtual environment:

    ```bash
    python -m venv myenv
    myenv\Scripts\activate -  #Windows
    source myenv/bin/activate - #Linux/ MacOs

3. Install dependencies:

    ```bash
    pip install -r requirements.txt

## Usage

1. Run the backend FastAPI server:

    ```bash
    uvicorn backend.main:app --reload

2. Run the frontend Streamlit app:

    ```bash
    streamlit run frontend/app.py

3. Enter the Job Role and Experience in the Streamlit UI to search and analyze resumes.



