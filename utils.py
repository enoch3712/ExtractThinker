import json

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