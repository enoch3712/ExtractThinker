from typing import Any, Optional
from pydantic import BaseModel
from extract_thinker.models.contract import Contract
import os


class Classification(BaseModel):
    name: str
    description: str
    contract: Optional[Contract] = None
    image: Optional[str] = None  # Path to the image file
    extractor: Optional[Any] = None

    def set_image(self, image_path: str):
        if os.path.isfile(image_path):
            self.image = image_path
        else:
            raise ValueError(f"The provided string '{image_path}' is not a valid file path.")
