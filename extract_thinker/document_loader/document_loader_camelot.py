"""Optional Camelot PDF table extraction."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from extract_thinker.document_loader.pdf_table_loader import PDFTableLoader


@dataclass
class CamelotConfig:
    flavor: str = 'lattice'
    password: Optional[str] = field(default=None, repr=False)
    cache_ttl: int = 300
    vision_enabled: bool = False
    read_pdf_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.cache_ttl <= 0:
            raise ValueError('cache_ttl must be positive')
        if self.flavor not in {'lattice', 'stream', 'network', 'hybrid', 'auto'}:
            raise ValueError('Unsupported Camelot flavor')
        if {'filepath', 'pages', 'password', 'flavor'} & self.read_pdf_kwargs.keys():
            raise ValueError('read_pdf_kwargs cannot override source, pages, password or flavor')


class DocumentLoaderCamelot(PDFTableLoader):
    def __init__(self, config: Optional[CamelotConfig] = None):
        try:
            import camelot
            if not callable(getattr(camelot, 'read_pdf', None)):
                raise ImportError('The installed camelot module is not camelot-py')
        except ImportError as exc:
            raise ImportError('Camelot loader requires `pip install camelot-py` (not the unrelated camelot package).') from exc
        self._camelot = camelot
        super().__init__(config or CamelotConfig())

    def _read_tables(self, path, pages):
        tables = self._camelot.read_pdf(path, pages='all', flavor=self.config.flavor,
                                      password=self.config.password, **self.config.read_pdf_kwargs)
        for table in tables:
            number = int(table.page)
            if not 1 <= number <= len(pages):
                raise ValueError(f'Camelot returned an invalid page number: {number}')
            rows = table.df.fillna('').astype(str).values.tolist()
            pages[number - 1]['tables'].append(rows)
