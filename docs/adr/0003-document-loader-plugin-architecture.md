# 3. Document loader plugin architecture

Date: 2024-06-09

## Status

Accepted

## Context

ExtractThinker needs to support multiple document types and OCR backends (Tesseract, EasyOCR, etc.). We want to make it easy to add new loaders without modifying the core extractor logic.

## Decision

We will use a plugin-like architecture for document loaders:
- Each loader implements a common interface (`DocumentLoader`).
- Loaders are registered and discovered by the extractor at runtime.
- Loader configuration is handled via config classes (e.g., `EasyOCRConfig`).

## Consequences

- New loaders can be added with minimal changes to the core.
- Loader-specific dependencies are isolated.
- The extractor can select the appropriate loader based on file type or capabilities.