class ExtractThinkerError(Exception):
    """Base exception class for ExtractThinker."""
    pass

class VisionError(ExtractThinkerError):
    """Base class for vision-related errors."""
    pass

class InvalidVisionDocumentLoaderError(VisionError):
    """Document loader does not support vision features."""
    pass