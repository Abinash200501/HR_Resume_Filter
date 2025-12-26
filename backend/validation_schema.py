from typing import Annotated
from pydantic import BaseModel, StringConstraints

class SearchRequest(BaseModel):
    job_role: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z\s]+$")]
    experience: Annotated[str, StringConstraints(pattern=r"^\d+(\.\d+)? years$")]
