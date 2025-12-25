import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import streamlit as st
import requests


st.title("HR Resume Filtering")

job_role = st.text_input("Enter the job description", placeholder="Gen AI Engineer")
experience = st.text_input("Years of experience", placeholder="3 years")

uploaded_files = st.file_uploader(
    "Upload Resumes", accept_multiple_files=True, type="pdf"
)

BACKEND_URL = "http://localhost:8000"

if st.button("Upload Files"):
    if not job_role or not experience:
        st.warning("Please fill in all mandatory fields!")
    elif not uploaded_files:
        st.warning("Please upload at least 1 file!")
    else:
        with st.spinner("Files are uploading..."):
            files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
            response = requests.post(f"{BACKEND_URL}/upload", files=files)
            if response.status_code == 200:
                st.success("Files are saved successfully")
            else:
                st.error(f"Upload failed: {response.text}")

    

if st.button("Find Best Resumes"):
    if not job_role or not experience:
        st.warning("Please fill in all mandatory fields!")
    elif not uploaded_files:
        st.warning("Please upload at least 1 file!")
    else:
        with st.spinner("Searching...."):
            response = requests.post(f"{BACKEND_URL}/search-and-analyse", json={"job_role": job_role, "experience": experience})
            if response.status_code == 200:
                st.success("Searching is done wait for few mins till we find the best resumes")
                output = response.json()
                llm_content = output.get('llm_response', {}).get("content", {})

                st.markdown(llm_content)
            else:
                st.error(f"Internal error: {response.text}")
        


