import json
from typing import Optional, Any, Union, get_origin, get_args
from pydantic import BaseModel, create_model

def remove_json_format(json_string: str) -> str:
    replace = json_string.replace("```json", "")
    replace = replace.replace("```", "")
    return replace.strip()


def remove_last_element(json_string: str) -> str:
    try:
        json.loads(json_string)
        return json_string
    except json.JSONDecodeError:
        pass

    last_index = json_string.rfind("},")

    if last_index == -1:
        return json_string

    trimmed_string = json_string[:last_index + 1]
    trimmed_string += ","
    return trimmed_string

def make_all_fields_optional(model: Any) -> Any:
    """Convert all fields of a Pydantic model to Optional."""
    if not issubclass(model, BaseModel):
        raise ValueError("model must be a subclass of BaseModel")

    fields = {}
    for field_name, field in model.model_fields.items():
        annotation = field.annotation
        # Check if field is already optional
        if get_origin(annotation) is not Union or type(None) not in get_args(annotation):
            fields[field_name] = (Optional[field.annotation], None)
        else:
            fields[field_name] = (field.annotation, None)

    NewModel = create_model(
        model.__name__ + "Optional",
        __base__=model,
        **fields
    )
    return NewModel