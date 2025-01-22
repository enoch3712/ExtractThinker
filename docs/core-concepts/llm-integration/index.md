# LLM Integration

!!! warning "🚧 In Development"
    This component is currently under active development. The API might change in future releases.

The LLM component in ExtractThinker acts as a bridge between your document processing pipeline and various Language Model providers. It handles request formatting, response parsing, and provider-specific optimizations.

<div align="center">
  <img src="../../assets/llm_image.png" alt="LLM Architecture" width="50%">
</div>

The architecture consists of:

- **Parser**: Uses [instructor](https://github.com/jxnl/instructor) for structured outputs with Pydantic

- **LLM Broker**: Leverages [litellm](https://github.com/BerriAI/litellm) for unified model interface

??? example "Base LLM Implementation"
    ```python
    --8<-- "extract_thinker/llm.py"
    ```

## Basic Usage

```python
from extract_thinker import LLM

# Initialize with specific model
llm = LLM("gpt-4o")
```