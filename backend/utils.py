import pymupdf
from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"

def read_files():

    results = []

    for index, file in enumerate(RESOURCE_DIR.glob("*.pdf")):
        pdf_bytes = file.read_bytes()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for i in doc:
            text += i.get_text()
        results.append({"file_name": file.name, 
                        "text": text, 
                        "resume_id": f"resume_{index}"})

    
    return results