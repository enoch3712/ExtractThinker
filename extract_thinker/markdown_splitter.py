"""Split Markdown by real headings, or classify Markdown pages with an LLM."""
from typing import Any, Dict, List, Optional

from extract_thinker.models.classification import Classification
from extract_thinker.text_splitter import TextSplitter


class MarkdownSplitter(TextSplitter):
    """Preserve Markdown structure when splitting sections or document pages.

    `split_sections` is deterministic and needs no model. Process eager/lazy
    document classification requires a model, like TextSplitter.
    """

    def __init__(self, model: Optional[str] = None, heading_level: int = 2):
        if isinstance(heading_level, bool) or not isinstance(heading_level, int) or not 1 <= heading_level <= 6:
            raise ValueError("heading_level must be an integer from 1 to 6")
        self.heading_level = heading_level
        self.model = model
        self.llm = None
        if model is not None:
            super().__init__(model)

    def split_sections(self, markdown: str) -> List[Dict[str, Any]]:
        """Split at headings up to heading_level; preserve all original text.

        Line ranges are one-based and inclusive. Heading-like text inside code
        fences is not a boundary. Both ATX and Setext headings are supported.
        """
        try:
            from markdown_it import MarkdownIt
        except ImportError as exc:
            raise ImportError("Markdown section splitting requires `pip install markdown-it-py`.") from exc
        if not isinstance(markdown, str):
            raise TypeError("markdown must be a string")
        if not markdown:
            return []
        tokens = MarkdownIt().parse(markdown)
        boundaries = []
        for index, token in enumerate(tokens):
            if token.type == "heading_open" and int(token.tag[1:]) <= self.heading_level:
                boundaries.append((token.map[0], tokens[index + 1].content, int(token.tag[1:])))
        lines = markdown.splitlines(keepends=True)
        if not boundaries or boundaries[0][0] != 0:
            boundaries.insert(0, (0, None, 0))
        sections = []
        for index, (start, heading, level) in enumerate(boundaries):
            end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(lines)
            sections.append({"content": "".join(lines[start:end]), "heading": heading,
                             "level": level, "start_line": start + 1, "end_line": end})
        return sections

    def _markdown_pages(self, pages):
        if self.llm is None:
            raise ValueError("A model is required for document classification; use split_sections for local splitting")
        return [dict(page, content=page.get("markdown", page.get("content", ""))) for page in pages]

    def split_eager_doc_group(self, document: List[dict], classifications: List[Classification]):
        return super().split_eager_doc_group(self._markdown_pages(document), classifications)

    def split_lazy_doc_group(self, document: List[dict], classifications: List[Classification]):
        pages = self._markdown_pages(document)
        if len(pages) < 2:
            from extract_thinker.models.doc_group import DocGroups, DocGroup
            result = DocGroups()
            if pages:
                groups = super().split_eager_doc_group(pages, classifications)
                result.doc_groups = [DocGroup(group.pages, group.classification, group.classification_id) for group in groups]
            return result
        return super().split_lazy_doc_group(pages, classifications)
