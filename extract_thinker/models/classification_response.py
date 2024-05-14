from pydantic import BaseModel


class ClassificationResponse(BaseModel):
    name: str

    def __hash__(self):
        return hash((self.name))
