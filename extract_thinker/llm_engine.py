from enum import Enum


class LLMEngine(Enum):
    """Supported LLM backends.
    
    Attributes:
        DEFAULT: Uses litellm + instructor for model interfacing and structured outputs
        PYDANTIC_AI: Uses pydantic-ai for enhanced Pydantic model integration
    """
    DEFAULT = "default"  # Default backend using litellm + instructor
    PYDANTIC_AI = "pydantic_ai"  # Pydantic AI backend for enhanced model integration