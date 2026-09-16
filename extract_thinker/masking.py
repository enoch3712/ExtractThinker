"""Local, reversible text masking before model extraction."""
import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional
from uuid import uuid4

from extract_thinker.document_loader.document_loader import DocumentLoader

EMAIL_PATTERN = r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w-])"


def _walk(value, transform):
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, list):
        return [_walk(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(_walk(item, transform) for item in value)
    if isinstance(value, dict):
        return {key: _walk(item, transform) for key, item in value.items()}
    return value


def _restore(value, mapping):
    if not mapping:
        return _walk(value, lambda text: text)
    pattern = re.compile('|'.join(re.escape(token) for token in mapping))
    return _walk(value, lambda text: pattern.sub(lambda match: mapping[match.group()], text))


@dataclass(frozen=True)
class MaskedContent:
    content: Any
    mapping: Dict[str, str] = field(repr=False)

    def restore(self, value):
        """Restore exact placeholders in string values of plain nested data."""
        return _restore(value, self.mapping)


class MaskingSession:
    """Reuse entity identities across pages/files in one extraction job."""
    def __init__(self, rules):
        self._rules = rules
        self._namespace = uuid4().hex
        self._values = {}
        self._mapping = {}
        self._lock = RLock()

    def _mask_text(self, text):
        if f'__ET_{self._namespace}_' in text:
            raise ValueError('Input contains a token from this masking session; pass original content')
        candidates = []
        for label, pattern in self._rules:
            for match in pattern.finditer(text):
                if match.start() == match.end():
                    raise ValueError('Masking patterns must not match empty spans')
                candidates.append((match.start(), match.end(), label))
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        parts, position = [], 0
        for start, end, label in candidates:
            if start < position:
                continue
            original = text[start:end]
            key = (label, original)
            if key not in self._values:
                token = f'__ET_{self._namespace}_{label}_{len(self._values) + 1}__'
                self._values[key] = token
                self._mapping[token] = original
            parts.extend((text[position:start], self._values[key]))
            position = end
        parts.append(text[position:])
        return ''.join(parts)

    def mask(self, value):
        with self._lock:
            return MaskedContent(_walk(value, self._mask_text), dict(self._mapping))

    def restore(self, value):
        with self._lock:
            return _restore(value, self._mapping)


class EntityMasker:
    """Mask explicitly supplied entities and regex matches locally.

    Built-in email matching is optional. Names, addresses and other entity types
    require supplied values or patterns; this is not a complete PII detector.
    """
    def __init__(self, entities: Optional[Dict[str, List[str]]] = None,
                 patterns: Optional[Dict[str, str]] = None, detect_emails: bool = True):
        rules = []
        if detect_emails:
            rules.append(('EMAIL', re.compile(EMAIL_PATTERN)))
        for label, values in (entities or {}).items():
            self._validate_label(label)
            if not isinstance(values, (list, tuple)) or not all(isinstance(value, str) and value for value in values):
                raise ValueError('Each entity label requires a list of nonempty strings')
            if values:
                # Longest alternatives prevent a short name masking only the
                # beginning of a longer name at the same source position.
                expression = '|'.join(re.escape(value) for value in sorted(set(values), key=len, reverse=True))
                rules.append((label, re.compile(r'(?<!\w)(?:' + expression + r')(?!\w)')))
        for label, expression in (patterns or {}).items():
            self._validate_label(label)
            pattern = re.compile(expression)
            if pattern.search(''):
                raise ValueError('Masking patterns must not match empty spans')
            rules.append((label, pattern))
        self._rules = tuple(rules)

    @staticmethod
    def _validate_label(label):
        if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', label):
            raise ValueError('Entity labels must contain letters, digits or underscores and start with a letter')

    def session(self):
        return MaskingSession(self._rules)

    def mask(self, value):
        """Mask one document with a fresh token namespace and return its mapping."""
        return self.session().mask(value)


class DocumentLoaderMasked(DocumentLoader):
    """Mask loaded text values before extraction; image input is unsupported.

    A wrapper owns one session across load calls, allowing multi-file extraction
    and restoration. Create a wrapper per job or call reset() between jobs.
    """
    def __init__(self, loader: DocumentLoader, masker: EntityMasker):
        super().__init__()
        self.loader = loader
        self.masker = masker
        self._session = masker.session()
        self.SUPPORTED_FORMATS = getattr(loader, 'SUPPORTED_FORMATS', [])

    def can_handle(self, source):
        return self.loader.can_handle(source)

    def can_handle_vision(self, source):
        return False

    def can_handle_paginate(self, source):
        return self.loader.can_handle_paginate(source)

    def set_vision_mode(self, enabled=True):
        if enabled:
            raise ValueError('Text masking does not redact image pixels; vision is not supported')
        super().set_vision_mode(False)

    def load(self, source):
        previous = self.loader.vision_mode
        try:
            self.loader.set_vision_mode(False)
            pages = self.loader.load(source)
        finally:
            self.loader.set_vision_mode(previous)
        if not isinstance(pages, list) or not all(isinstance(page, dict) for page in pages):
            raise ValueError('Masking requires a loader returning page dictionaries')
        # Even cached/preloaded parsers can return images while vision is off.
        clean_pages = [{key: value for key, value in page.items() if key not in ('image', 'images')}
                       for page in pages]
        return self._session.mask(clean_pages).content

    def restore(self, value):
        return self._session.restore(value)

    def reset(self):
        self._session = self.masker.session()
