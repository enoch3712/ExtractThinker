import base64
import json
import re
import yaml
from PIL import Image
import tiktoken
from pydantic import BaseModel
import typing
import os


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_type(image_path):
    try:
        img = Image.open(image_path)
        return img.format.lower()
    except IOError as e:
        return f"An error occurred: {str(e)}"


def verify_json(json_content: str):
    try:
        data = json.loads(json_content)
        return True, data, "JSON is valid."
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {str(e)}"


def convert_json_to_yaml(json_data: str):
    yaml_content = yaml.safe_dump(json_data, default_flow_style=False, sort_keys=False)
    return yaml_content


def verify_yaml(yaml_content: str):
    try:
        data = yaml.safe_load(yaml_content)
        return True, data, "YAML is valid."
    except yaml.YAMLError as e:
        return False, None, f"Invalid YAML: {str(e)}"


def convert_yaml_to_json(yaml_data: str):
    json_content = json.dumps(yaml_data, indent=4)
    return json_content


def num_tokens_from_string(text: str) -> int:
    """
    Returns the number of tokens in a text string using the GPT-4 encoding.

    Args:
    text (str): The text string to tokenize.

    Returns:
    int: The number of tokens.
    """
    # Automatically load the encoding for GPT-4
    encoding = tiktoken.encoding_for_model("gpt-4")

    # Encode the string to get the list of token integers.
    tokens = encoding.encode(text)

    # Return the number of tokens.
    return len(tokens)


def string_to_pydantic_class(class_definition: str):
    # Define the namespace where the class will be evaluated
    namespace = {
        'BaseModel': BaseModel,
        'typing': typing
    }
    # Evaluate the class definition in the given namespace
    pydantic_class = eval(class_definition, namespace)
    return pydantic_class


def extract_json(text):
    # Find the JSON string in the text
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        json_str = match.group()
        try:
            # Try to load the JSON string
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError:
            print("Invalid JSON")
            return None
    else:
        print("No JSON found")
        return None


def get_file_extension(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext[1:]  # remove the dot
    return ext


def json_to_formatted_string(data):
    result = []
    for sheet, rows in data.items():
        result.append(f"##{sheet}")
        for row in rows:
            result.append(','.join(map(str, row)))
    return '\n'.join(result)