from pydantic import BaseModel

class SearchRequest(BaseModel):
    job_role: str
    experience: str