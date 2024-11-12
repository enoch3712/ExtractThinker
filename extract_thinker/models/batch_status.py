from typing import Optional
from pydantic import BaseModel

class BatchStatus(BaseModel):
    id: str
    status: str
    output_file_id: Optional[str] = None