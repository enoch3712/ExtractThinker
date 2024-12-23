import base64
import json
import re
import yaml
from PIL import Image
from pydantic import BaseModel
import typing
import os
from io import BytesIO
import sys
from typing import Optional, Any, Union, get_origin, get_args, List, Dict
from pydantic import BaseModel, create_model

def encode_image(image_source: Union[str, BytesIO, bytes, Image.Image]) -> str:
    """
    Encode an image to base64 string from various sources.

    Args:
        image_source (Union[str, BytesIO, bytes, PIL.Image.Image]): The image source.

    Returns:
        str: Base64 encoded string of the image
    """
    try:
        if isinstance(image_source, str):
            with open(image_source, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(image_source, BytesIO):
            current_position = image_source.tell()
            image_source.seek(0)
            encoded = base64.b64encode(image_source.read()).decode("utf-8")
            image_source.seek(current_position)
            return encoded
        elif isinstance(image_source, bytes):
            return base64.b64encode(image_source).decode("utf-8")
        elif isinstance(image_source, Image.Image):
            buffer = BytesIO()
            image_source.save(buffer, format='JPEG')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")
        else:
            raise ValueError("Image source must be a file path (str), BytesIO stream, bytes, or PIL Image")
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


def json_to_formatted_string(data: dict) -> str:
    """
    Converts JSON/dict data into a formatted string representation.
    Particularly useful for spreadsheet data.
    
    Args:
        data: Dictionary containing the data to format
        
    Returns:
        A formatted string representation of the data
    """
    if not data:
        return ""
        
    # For spreadsheet-like data, create a tabular format
    if isinstance(data, dict):
        # Convert to yaml for readable output
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    return str(data)

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

def add_classification_structure(response_model: type[BaseModel], indent: int = 1) -> str:
    """
    Creates a string representation of the model's structure to help guide LLM responses.
    Handles nested Pydantic models recursively.
    
    Args:
        response_model: The Pydantic model to analyze
        indent: Current indentation level (for recursive calls)
        
    Returns:
        str: A formatted string describing the model structure
    """
    content = "\tResponse Structure:\n" if indent == 1 else ""
    tab = "\t" * indent

    # Iterate over the fields of the model
    for name, field in response_model.model_fields.items():
        # Extract the type and required status
        field_str = str(field)
        field_type = field.annotation
        required = 'required' in field_str

        # Get base type string
        type_str = str(field_type)
        if hasattr(field_type, "__name__"):
            type_str = field_type.__name__

        # Handle container types (List, Dict, etc.)
        origin = get_origin(field_type)
        if origin is not None:
            args = get_args(field_type)
            if origin in (list, List):
                type_str = f"List[{args[0].__name__ if hasattr(args[0], '__name__') else str(args[0])}]"
            elif origin in (dict, Dict):
                key_type = args[0].__name__ if hasattr(args[0], '__name__') else str(args[0])
                value_type = args[1].__name__ if hasattr(args[1], '__name__') else str(args[1])
                type_str = f"Dict[{key_type}, {value_type}]"

        # Add the field details
        field_details = f"{tab}Name: {name}, Type: {type_str}, Attributes: required={required}"
        content += field_details + "\n"

        # Recursively handle nested Pydantic models
        if hasattr(field_type, "model_fields"):
            # It's a nested Pydantic model
            content += f"{tab}Nested structure for {name}:\n"
            content += add_classification_structure(field_type, indent + 1)
        elif origin in (list, List):
            # Check if list contains a Pydantic model
            item_type = args[0]
            if hasattr(item_type, "model_fields"):
                content += f"{tab}Nested structure for {name} items:\n"
                content += add_classification_structure(item_type, indent + 1)
        elif origin in (dict, Dict):
            # Check if dict value type is a Pydantic model
            value_type = args[1]
            if hasattr(value_type, "model_fields"):
                content += f"{tab}Nested structure for {name} values:\n"
                content += add_classification_structure(value_type, indent + 1)

    return content

MIME_TYPE_MAPPING = {
    # Documents
    'pdf': 'application/pdf',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'rtf': 'application/rtf',
    'txt': 'text/plain',
    'odt': 'application/vnd.oasis.opendocument.text',
    'tex': 'application/x-tex',
    'markdown': ['text/markdown', 'text/x-markdown'],
    'md': ['text/markdown', 'text/x-markdown'],
    
    # Spreadsheets
    'xls': 'application/vnd.ms-excel',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'ods': 'application/vnd.oasis.opendocument.spreadsheet',
    'csv': ['text/csv', 'application/csv'],
    'tsv': 'text/tab-separated-values',
    
    # Presentations
    'ppt': 'application/vnd.ms-powerpoint',
    'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'odp': 'application/vnd.oasis.opendocument.presentation',
    'key': 'application/vnd.apple.keynote',
    
    # Images
    'jpg': ['image/jpeg', 'image/jpg'],
    'jpeg': ['image/jpeg', 'image/jpg'],
    'png': 'image/png',
    'gif': 'image/gif',
    'bmp': 'image/bmp',
    'tiff': 'image/tiff',
    'tif': 'image/tiff',
    'webp': 'image/webp',
    'svg': ['image/svg+xml', 'application/svg+xml'],
    'ico': 'image/x-icon',
    'raw': 'image/x-raw',
    'heic': 'image/heic',
    'heif': 'image/heif',
    
    # Web
    'html': ['text/html', 'application/xhtml+xml'],
    'htm': ['text/html', 'application/xhtml+xml'],
    'xhtml': 'application/xhtml+xml',
    'xml': ['application/xml', 'text/xml'],
    'json': 'application/json',
    'yaml': ['application/yaml', 'text/yaml'],
    'yml': ['application/yaml', 'text/yaml'],
    
    # Archives
    'zip': 'application/zip',
    'rar': 'application/x-rar-compressed',
    '7z': 'application/x-7z-compressed',
    'tar': 'application/x-tar',
    'gz': 'application/gzip',
    
    # Audio
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
    'ogg': 'audio/ogg',
    'flac': 'audio/flac',
    'm4a': 'audio/mp4',
    'aac': 'audio/aac',
    
    # Video
    'mp4': 'video/mp4',
    'avi': 'video/x-msvideo',
    'mkv': 'video/x-matroska',
    'mov': 'video/quicktime',
    'wmv': 'video/x-ms-wmv',
    'flv': 'video/x-flv',
    'webm': 'video/webm',
    
    # Ebooks
    'epub': 'application/epub+zip',
    'mobi': 'application/x-mobipocket-ebook',
    'azw': 'application/vnd.amazon.ebook',
    'azw3': 'application/vnd.amazon.ebook',
    
    # CAD and 3D
    'dwg': 'application/acad',
    'dxf': 'application/dxf',
    'stl': 'model/stl',
    'obj': 'model/obj',
    
    # Fonts
    'ttf': 'font/ttf',
    'otf': 'font/otf',
    'woff': 'font/woff',
    'woff2': 'font/woff2',
    
    # Programming
    'py': 'text/x-python',
    'js': 'text/javascript',
    'css': 'text/css',
    'java': 'text/x-java-source',
    'cpp': 'text/x-c++src',
    'c': 'text/x-c',
    'swift': 'text/x-swift',
    'go': 'text/x-go',
    'rs': 'text/x-rust',
    
    # Database
    'sql': 'application/sql',
    'db': 'application/x-sqlite3',
    'sqlite': 'application/x-sqlite3',
    
    # Email
    'eml': 'message/rfc822',
    'msg': 'application/vnd.ms-outlook',
    
    # Scientific/Technical
    'nb': 'application/mathematica',
    'mat': 'application/x-matlab-data',
    'r': 'text/x-r',
    'tex': 'application/x-tex',
    
    # Configuration
    'ini': 'text/plain',
    'conf': 'text/plain',
    'toml': 'application/toml',
    
    # Vector Graphics
    'ai': 'application/postscript',
    'eps': 'application/postscript',
    'ps': 'application/postscript',
}

def check_mime_type(mime: str, supported_formats: List[str]) -> bool:
    """
    Check if a MIME type matches any of the supported formats.
    
    Args:
        mime: The MIME type to check
        supported_formats: List of supported format extensions
        
    Returns:
        bool: True if the MIME type matches any supported format
    """
    for fmt in supported_formats:
        expected_mime = MIME_TYPE_MAPPING.get(fmt.lower())
        if expected_mime:
            if isinstance(expected_mime, list):
                if mime in expected_mime:
                    return True
            elif mime == expected_mime:
                return True
    return False