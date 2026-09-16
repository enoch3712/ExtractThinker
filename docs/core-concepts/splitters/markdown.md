# Markdown splitter

`MarkdownSplitter` offers local heading-based section splitting and Markdown-aware
LLM document grouping. It is available on `main` after release 0.1.14.

## Local section splitting

Install `markdown-it-py`, then split without a model or API key:

```python
from extract_thinker import MarkdownSplitter

sections = MarkdownSplitter(heading_level=2).split_sections(
    "# Invoice\nHeader\n\n## Items\n| Item | Price |\n| --- | --- |\n| Pen | 2 |\n"
)
for section in sections:
    print(section["heading"], section["start_line"], section["end_line"])
    print(section["content"])
```

The splitter recognizes ATX (`#`) and Setext (underlined) headings using the
Markdown parser. Heading-like text inside code fences is left intact. It splits
at headings at or above the configured level (1–6), keeps any preamble, and
preserves all original text and newlines. Line ranges are one-based and inclusive.
Joining each section's `content` reconstructs the input exactly.

## Document grouping with Process

```python
from extract_thinker import MarkdownSplitter

splitter = MarkdownSplitter(model="your-provider/your-model")
process.load_splitter(splitter)
```

This uses the existing eager/lazy semantic page-grouping strategies and requires
an LLM. When the loader provides a `markdown` field, the splitter uses it in
preference to plain `content`, preserving headings, lists and tables in the
classification prompt. A model-free splitter can only use `split_sections`.
Section boundaries within one page are separate from Process's page-based
classification groups; use the section dictionaries with `DocumentLoaderData`
when you want to extract individual sections.
