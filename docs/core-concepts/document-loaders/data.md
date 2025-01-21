# Data Document Loader

The Data loader is a specialized loader that handles pre-processed data in a standardized format. It provides caching support and vision mode compatibility.

## Supported Format

The loader expects data in the following standard format:
```python
[
  {
    "content": "...some text...",
    "image": None or [] or bytes
  }
]
```

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderData

# Initialize with default settings
loader = DocumentLoaderData()

# Load pre-formatted data
data = [{"content": "Sample text", "image": None}]
pages = loader.load(data)

# Process content
for page in pages:
    # Access text content
    text = page["content"]
    # Access image data if present
    image = page["image"]
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderData, DataLoaderConfig

# Create configuration
config = DataLoaderConfig(
    content=None,                # Initial content
    cache_ttl=600,              # Cache results for 10 minutes
    supports_vision=True         # Enable vision support
)

# Initialize loader with configuration
loader = DocumentLoaderData(config)

# Load and process content
pages = loader.load("raw text content")
```

## Configuration Options

The `DataLoaderConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `supports_vision` | bool | True | Whether vision mode is supported |