"""Optional native SDK checks; Tabula additionally needs Java on PATH."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import pymupdf
from extract_thinker import DocumentLoaderCamelot, CamelotConfig, DocumentLoaderTabula, TabulaConfig


@pytest.fixture
def ruled_pdf(tmp_path):
    path = tmp_path / 'table.pdf'
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        for y in (50, 80, 110, 140):
            page.draw_line((50, y), (300, y))
        for x in (50, 180, 300):
            page.draw_line((x, 50), (x, 140))
        for row, cells in enumerate([['Item', 'Amount'], ['Book', '25'], ['Pen', '5']]):
            for column, text in enumerate(cells):
                page.insert_text((60 + column * 130, 70 + row * 30), text, fontsize=12)
        pdf.new_page()
        pdf.save(path)
    return path


def test_camelot_sdk_extracts_real_table_and_preserves_blank_page(ruled_pdf):
    pytest.importorskip('camelot')
    pages = DocumentLoaderCamelot(CamelotConfig(flavor='lattice')).load(str(ruled_pdf))
    assert len(pages) == 2
    assert pages[1]['tables'] == []
    rows = pages[0]['tables'][0]
    assert ['Book', '25'] in rows
    assert rows[0] == ['Item', 'Amount']


def test_tabula_sdk_extracts_real_table_and_preserves_blank_page(ruled_pdf, monkeypatch):
    pytest.importorskip('tabula')
    # Optional portable test runtime; production users provide their own Java.
    try:
        import jdk4py
        monkeypatch.setenv('PATH', str(jdk4py.JAVA_HOME / 'bin') + os.pathsep + os.environ['PATH'])
    except ImportError:
        pass
    java = shutil.which('java')
    if java is None or subprocess.run([java, '-version'], capture_output=True).returncode:
        pytest.skip('Java runtime not available')
    pages = DocumentLoaderTabula(TabulaConfig(lattice=True, read_pdf_kwargs={'force_subprocess': True})).load(str(ruled_pdf))
    assert len(pages) == 2
    assert pages[1]['tables'] == []
    assert ['Book', '25'] in pages[0]['tables'][0]
    assert pages[0]['tables'][0][0] == ['Item', 'Amount']
