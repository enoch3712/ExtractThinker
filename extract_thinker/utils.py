import base64
import json
import re
import yaml
from PIL import Image
import tiktoken
from pydantic import BaseModel
import typing
import os
from io import BytesIO
from typing import Union
import sys
from typing import List

def encode_image(image_source: Union[str, BytesIO]) -> str:
    """
    Encode an image to base64 string from either a file path or BytesIO stream.

    Args:
        image_source (Union[str, BytesIO]): The image source, either a file path or BytesIO stream

    Returns:
        str: Base64 encoded string of the image
    """
    try:
        if isinstance(image_source, str):
            with open(image_source, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(image_source, BytesIO):
            # Save current position
            current_position = image_source.tell()
            # Move to start of stream
            image_source.seek(0)
            # Encode stream content
            encoded = base64.b64encode(image_source.read()).decode("utf-8")
            # Restore original position
            image_source.seek(current_position)
            return encoded
        else:
            raise ValueError("Image source must be either a file path (str) or BytesIO stream")
    except Exception as e:
        raise Exception(f"Failed to encode image: {str(e)}")

def is_pdf_stream(stream: Union[BytesIO, str]) -> bool:
    """
    Checks if the provided stream is a PDF.

    Args:
        stream (Union[BytesIO, str]): The stream to check. It can be a BytesIO object or a file path as a string.

    Returns:
        bool: True if the stream is a PDF, False otherwise.
    """
    try:
        if isinstance(stream, BytesIO):
            # Save the current position
            current_position = stream.tell()
            # Move to the start of the stream
            stream.seek(0)
            # Read the first 5 bytes to check the PDF signature
            header = stream.read(5)
            # Restore the original position
            stream.seek(current_position)
        elif isinstance(stream, str):
            if os.path.isfile(stream):
                with open(stream, 'rb') as file:
                    header = file.read(5)
            else:
                # If it's not a file path, assume it's a bytes string
                header = stream.encode()[:5] if isinstance(stream, str) else b''
        else:
            # Unsupported type
            return False

        # PDF files start with '%PDF-'
        return header == b'%PDF-'
    except Exception as e:
        # Optional: Log the exception if logging is set up
        # logger.error(f"Error checking if stream is PDF: {e}")
        return False

def get_image_type(source):
    try:
        if isinstance(source, str):
            img = Image.open(source)
        elif isinstance(source, BytesIO):
            source.seek(0)
            img = Image.open(source)
            source.seek(0)
        else:
            return None
        return img.format.lower()
    except IOError as e:
        return None

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


def simple_token_counter(text: str) -> int:
    """
    A lightweight token counter that approximates GPT tokenization rules.
    Used as fallback for Python 3.13+
    """
    if not text:
        return 0
        
    # Preprocessing
    text = text.lower()
    
    # Split into chunks (words, punctuation, numbers)
    chunks = re.findall(r"""
        [a-z]{1,20}|         # Words
        [0-9]+|              # Numbers
        [^a-z0-9\s]{1,2}|   # 1-2 special chars
        \s+                  # Whitespace
        """, text, re.VERBOSE)
    
    # Count tokens with some basic rules
    token_count = 0
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # Whitespace
        if chunk.isspace():
            token_count += 1
            continue
            
        # Short words (likely one token)
        if len(chunk) <= 4:
            token_count += 1
            continue
            
        # Longer words (may be split into subwords)
        token_count += max(1, len(chunk) // 4)
    
    return token_count


def num_tokens_from_string(text: str) -> int:
    """
    Returns the number of tokens in a text string.
    Uses tiktoken for Python <3.13, falls back to simple counter for 3.13+
    """
    python_version = sys.version_info[:2]
    
    # For Python versions below 3.13, use tiktoken
    if python_version < (3, 13):
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(text))
        except ImportError:
            print("Warning: tiktoken not installed for Python <3.13. Using fallback counter.")
            return simple_token_counter(text)
    
    # For Python 3.13+, use the simple counter
    return simple_token_counter(text)


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
    if isinstance(file_path, str):
        _, ext = os.path.splitext(file_path)
        ext = ext[1:]  # remove the dot
        return ext.lower()
    else:
        return None


def json_to_formatted_string(data):
    result = []
    for sheet, rows in data.items():
        result.append(f"##{sheet}")
        for row in rows:
            result.append(','.join(map(str, row)))
    return '\n'.join(result)