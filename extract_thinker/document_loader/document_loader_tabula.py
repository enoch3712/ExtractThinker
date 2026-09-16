"""Optional tabula-py PDF table extraction (requires a Java runtime)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from extract_thinker.document_loader.pdf_table_loader import PDFTableLoader


@dataclass
class TabulaConfig:
    lattice: bool = False
    stream: bool = False
    guess: bool = True
    password: Optional[str] = field(default=None, repr=False)
    cache_ttl: int = 300
    vision_enabled: bool = False
    java_options: Optional[List[str]] = None
    read_pdf_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.cache_ttl <= 0:
            raise ValueError('cache_ttl must be positive')
        if self.lattice and self.stream:
            raise ValueError('Choose lattice or stream, not both')
        reserved = {'input_path', 'pages', 'password', 'output_format', 'multiple_tables',
                    'lattice', 'stream', 'guess', 'java_options', 'output_path', 'batch', 'format'}
        if reserved & self.read_pdf_kwargs.keys():
            raise ValueError('read_pdf_kwargs cannot override loader input/output or explicit settings')


class DocumentLoaderTabula(PDFTableLoader):
    def __init__(self, config: Optional[TabulaConfig] = None):
        try:
            import tabula
            if not callable(getattr(tabula, 'read_pdf', None)):
                raise ImportError('The installed tabula module is not tabula-py')
        except ImportError as exc:
            raise ImportError('Tabula loader requires `pip install tabula-py` and a Java runtime on PATH.') from exc
        self._tabula = tabula
        super().__init__(config or TabulaConfig())

    def _read_tables(self, path, pages):
        # JSON tables do not consistently identify their source page. Request
        # each page explicitly so blanks and multiple tables cannot shift pages.
        for page in pages:
            tables = self._tabula.read_pdf(
                path, pages=page['page_number'], output_format='json',
                password=self.config.password, lattice=self.config.lattice,
                stream=self.config.stream, guess=self.config.guess,
                java_options=self.config.java_options, **self.config.read_pdf_kwargs,
            )
            for table in tables:
                page['tables'].append([
                    [str(cell['text']) if cell.get('text') is not None else '' for cell in row]
                    for row in table['data']
                ])
