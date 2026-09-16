from io import BytesIO
import pytest
from PIL import Image
from extract_thinker import DocumentLoaderPyMuPDF, PyMuPDFConfig, DocumentRegion

pymupdf = pytest.importorskip('pymupdf')


@pytest.fixture
def pdf_bytes():
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        page.insert_text((40, 80), 'First page invoice')
        page = document.new_page(width=300, height=400)
        page.insert_text((40, 80), 'Second page invoice')
        page.set_rotation(90)
        return document.tobytes()


def test_pymupdf_path_and_stream_text(tmp_path, pdf_bytes):
    path = tmp_path / 'invoice.pdf'
    path.write_bytes(pdf_bytes)
    loader = DocumentLoaderPyMuPDF()
    path_pages = loader.load(str(path))
    assert path_pages == loader.load(BytesIO(pdf_bytes))
    assert len(path_pages) == 2
    assert 'First page invoice' in path_pages[0]['content']
    assert 'Second page invoice' in path_pages[1]['content']
    assert 'image' not in path_pages[0]


def test_pymupdf_regions_align_with_rotated_render(pdf_bytes):
    loader = DocumentLoaderPyMuPDF(PyMuPDFConfig(include_bbox=True, vision_enabled=True, dpi=72))
    pages = loader.load(BytesIO(pdf_bytes))
    for page in pages:
        region = DocumentRegion.model_validate(page['regions'][0])
        assert region.bounding_box.page == page['page_number']
        assert region.text.strip() in page['content']
    with Image.open(BytesIO(pages[0]['image'])) as image:
        assert image.size == (300, 400)
    with Image.open(BytesIO(pages[1]['image'])) as image:
        assert image.size == (400, 300)
    first = pages[0]['regions'][0]['bounding_box']
    second = pages[1]['regions'][0]['bounding_box']
    assert first['x0'] == pytest.approx(40 / 300)
    assert second['y0'] == pytest.approx(40 / 300)
    assert second['x0'] > 0.7


def test_pymupdf_page_selection_and_vision_cache(pdf_bytes):
    loader = DocumentLoaderPyMuPDF()
    stream = BytesIO(pdf_bytes)
    assert loader.load_pages(stream, [2])[0]['page_number'] == 2
    loader.set_vision_mode(True)
    loader.set_max_image_size(100)
    pages = loader.load(stream)
    with Image.open(BytesIO(pages[0]['image'])) as image:
        assert max(image.size) == 100
    loader.set_vision_mode(False)
    assert 'image' not in loader.load(stream)[0]


def test_encrypted_pdf_requires_password():
    with pymupdf.open() as document:
        document.new_page()
        data = document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                                owner_pw='owner', user_pw='reader')
    with pytest.raises(ValueError, match='password'):
        DocumentLoaderPyMuPDF().load(BytesIO(data))
    pages = DocumentLoaderPyMuPDF(PyMuPDFConfig(password='reader')).load(BytesIO(data))
    assert len(pages) == 1


def test_blank_page_has_no_tables_or_regions():
    with pymupdf.open() as document:
        document.new_page()
        data = document.tobytes()
    loader = DocumentLoaderPyMuPDF(PyMuPDFConfig(extract_tables=True, include_bbox=True))
    page = loader.load(BytesIO(data))[0]
    assert page['content'] == ''
    assert page['tables'] == []
    assert page['regions'] == []


@pytest.mark.parametrize('dpi', [0, -1, True, 1.5])
def test_bad_dpi_rejected(dpi):
    with pytest.raises(ValueError, match='dpi'):
        PyMuPDFConfig(dpi=dpi)


def test_pymupdf_extracts_real_table():
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        for x in [40, 140, 240]:
            page.draw_line((x, 40), (x, 120))
        for y in [40, 80, 120]:
            page.draw_line((40, y), (240, y))
        for x, y, text in [(50, 60, 'Item'), (150, 60, 'Price'), (50, 100, 'Pen'), (150, 100, '2')]:
            page.insert_text((x, y), text)
        data = document.tobytes()
    loader = DocumentLoaderPyMuPDF(PyMuPDFConfig(extract_tables=True))
    assert loader.load(BytesIO(data))[0]['tables'] == [[['Item', 'Price'], ['Pen', '2']]]
