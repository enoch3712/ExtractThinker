"""Optional, model-download-free checks against the installed Docling SDK."""
import pytest

pytest.importorskip('docling')
from extract_thinker import DoclingConfig, DocumentLoaderDocling
from docling.datamodel.base_models import InputFormat


def test_sdk_configuration_and_html_conversion(tmp_path):
    config = DoclingConfig(table_structure_enabled=False, force_full_page_ocr=True)
    pipeline = config.format_options[InputFormat.PDF].pipeline_options
    assert pipeline.do_table_structure is False
    assert pipeline.ocr_options.force_full_page_ocr is True
    loader = DocumentLoaderDocling(config)
    path = tmp_path / 'document.html'
    path.write_text('<html><body><h1>Compatibility check</h1><p>Invoice ABC123</p></body></html>')
    pages = loader.load(str(path))
    assert 'ABC123' in pages[0]['content']
    assert pages[0]['markdown'] == pages[0]['content']
    assert pages[0]['page_number'] == 1
