# Splitters

In document processing, splitting enables the separation of individual documents or sections within a combined file. This task is especially crucial when handling batches of documents where different parts may need distinct processing, and always with Sonnet. This can be done with two strategies: Eager and Lazy.

<div align="center">
  <img src="../../assets/splitter_image.png" alt="Splitter Flow">
</div>

### Page-Level Processing

Splitters work at the page level, determining which pages belong together as a single document. For example:

- A 10-page PDF might contain three separate invoices

- A scanned document might contain multiple forms

- A batch of documents might need to be separated by document type

The challenge is determining where one document ends and another begins, which is where our splitting strategies come in.

## Eager vs. Lazy Approaches

Eager and Lazy splitting have distinct use cases based on document size and the complexity of relationships between pages.

### Eager Splitting

Eager splitting processes all pages in a single pass, identifying and dividing all sections at once. It's efficient for small to medium-sized documents where context size does not limit performance.

```python
from extract_thinker import Splitter, SplittingStrategy

splitter = Splitter()
result = splitter.split(
    document,
    strategy=SplittingStrategy.EAGER
)
```

Benefits of Eager Splitting:
- **Speed**: Faster processing since all split points are determined upfront
- **Simplicity**: Ideal for documents that fit entirely within the model's context window
- **Consistency**: Better for documents where relationships between pages are important

### Lazy Splitting

Lazy splitting processes pages incrementally in chunks, assessing smaller groups of pages at a time to decide if they belong together. In this use case, groups of two pages are processed and checked for continuity, allowing it to scale efficiently for larger documents.

```python
result = splitter.split(
    document,
    strategy=SplittingStrategy.LAZY
)
```

Benefits of Lazy Splitting:
- **Scalability**: Well-suited for documents that exceed the model's context window
- **Memory Efficiency**: Processes only what's needed when needed
- **Flexibility**: Better for streaming or real-time processing

??? example "Base Splitter Implementation"
    The base Splitter class provides both eager and lazy implementations:
    ```python
    --8<-- "extract_thinker/splitter.py"
    ```

## Available Splitters

ExtractThinker provides two main splitter implementations:

- [Text Splitter](text.md): For text-based document splitting
- [Image Splitter](image.md): For image-based document splitting

## Recommended Approach

For most IDP use cases, Eager Splitting is appropriate since it offers:
- Simpler implementation
- Better handling of page relationships
- Faster processing for typical document sizes (under 50 pages)

However, consider Lazy Splitting when:
- Processing very large documents (50+ pages)
- Working with limited memory
- Handling streaming document inputs

## Best Practices

- Choose strategy based on document size and page count
- Consider context window limitations of your LLM

## Reliable classification IDs

Text, image, and Markdown splitters assign one-based numeric IDs to classifications in the order supplied to each split request. The model selects IDs, and the returned groups contain both `classification` (the original display name) and `classification_id` (the numeric selection). `Process.extract()` uses the ID to select the original contract and extractor, so two classifications can share a display name without being confused.

```python
classifications = [
    Classification(name="Invoice", description="Sales invoice", contract=SalesInvoice, extractor=sales_extractor),
    Classification(name="Invoice", description="Purchase invoice", contract=PurchaseInvoice, extractor=purchase_extractor),
]
groups = splitter.split_eager_doc_group(pages, classifications)
for group in groups:
    print(group.pages, group.classification_id, group.classification)
```

IDs belong to the supplied classification list, not a global registry. Preserve that list's order when using the results outside `Process`. Legacy groups that contain only a name remain accepted when the name identifies exactly one classification; unknown or duplicate names raise an error.

Split failures now propagate instead of silently choosing the first classification or returning an `unknown` group. Eager splitting rejects omitted, duplicated, reordered, or out-of-range pages. Lazy splitting rejects conflicting classifications across overlapping page comparisons. These checks prevent silent routing errors; they do not guarantee that the model's semantic classification is correct. Single-page direct splitter calls perform classification, and empty input returns empty groups.
