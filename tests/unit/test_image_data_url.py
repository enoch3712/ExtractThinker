from io import BytesIO
import base64
import pytest
from PIL import Image
from extract_thinker.utils import image_to_data_url


@pytest.mark.parametrize('format,mime', [('PNG', 'image/png'), ('JPEG', 'image/jpeg'), ('WEBP', 'image/webp')])
def test_image_data_url_uses_actual_format(format, mime):
    stream = BytesIO()
    Image.new('RGB', (8, 8)).save(stream, format=format)
    encoded = image_to_data_url(stream.getvalue())
    assert encoded.startswith(f'data:{mime};base64,')
    assert base64.b64decode(encoded.split(',', 1)[1]) == stream.getvalue()


def test_pil_transparency_is_preserved():
    encoded = image_to_data_url(Image.new('RGBA', (8, 8), (1, 2, 3, 0)))
    assert encoded.startswith('data:image/png;base64,')
    with Image.open(BytesIO(base64.b64decode(encoded.split(',', 1)[1]))) as image:
        assert image.mode == 'RGBA'
