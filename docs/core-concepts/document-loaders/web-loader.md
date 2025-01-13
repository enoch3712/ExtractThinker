# Web Document Loader

The Web loader extracts content from web pages using BeautifulSoup. It supports HTML parsing, content cleaning, and custom element handling.

## Supported Formats

- html
- htm
- xhtml
- url

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderBeautifulSoup

# Initialize with default settings
loader = DocumentLoaderBeautifulSoup()

# Load document
pages = loader.load("https://example.com")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderBeautifulSoup, BeautifulSoupConfig

# Create configuration
config = BeautifulSoupConfig(
    header_handling="extract",     # Extract headers as separate content
    parser="lxml",                # Use lxml parser
    remove_elements=[             # Elements to remove
        "script", "style", "nav", "footer"
    ],
    max_tokens=8192,             # Maximum tokens per page
    request_timeout=30,          # Request timeout in seconds
    cache_ttl=600               # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderBeautifulSoup(config)

# Load and process document
pages = loader.load("https://example.com")
```

## Configuration Options

The `BeautifulSoupConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `header_handling` | str | "ignore" | How to handle headers |
| `parser` | str | "html.parser" | HTML parser to use |
| `remove_elements` | List[str] | None | Elements to remove |
| `max_tokens` | int | None | Maximum tokens per page |
| `request_timeout` | int | 10 | Request timeout in seconds |

## Features

- Web page content extraction
- Header handling options
- Custom element removal
- Multiple parser support
- Token limit control
- Request timeout control
- Caching support
- Stream-based loading

## Notes

- Vision mode is not supported
- Requires internet connection for URLs
- Local HTML files are supported
- Respects robots.txt
- May require custom headers for some sites
