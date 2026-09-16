import asyncio
import base64
import re
from io import BytesIO
from unittest.mock import Mock
import pytest
from PIL import Image
from extract_thinker import MarkdownConverter


def image_bytes(color='white', fmt='PNG'):
    data = BytesIO()
    Image.new('RGB', (2, 2), color).save(data, format=fmt)
    return data.getvalue()


def converter(pages, completion='Converted text'):
    loader = Mock()
    loader.load.return_value = pages
    llm = Mock()
    llm.raw_completion.return_value = completion
    return MarkdownConverter(loader, llm)


def test_all_images_embedded_per_page_with_correct_media_type_and_labels():
    png = image_bytes()
    jpeg = image_bytes('black', 'JPEG')
    md = converter([{'content': 'one', 'image': png},
                    {'content': 'two', 'images': [png, jpeg], 'image': png}])
    result = md.to_markdown('source', pages=[2, 1], include_images=True, vision=False)
    assert result[0].count('![Page 2 image') == 2
    assert 'data:image/png;base64,' in result[0]
    assert 'data:image/jpeg;base64,' in result[0]
    assert result[1].count('![Page 1 image') == 1
    for call in md.llm.raw_completion.call_args_list:
        assert 'image_url' not in str(call.kwargs['messages'])
    md.document_loader.set_vision_mode.assert_called_once_with(True)


def test_image_dictionary_base64_is_supported():
    data = image_bytes()
    md = converter([{'content': 'page', 'images': [{'base64': base64.b64encode(data).decode()}]}])
    assert 'data:image/png;base64,' in md.to_markdown('source', include_images=True)[0]


def test_images_not_embedded_by_default():
    md = converter([{'content': 'page', 'image': image_bytes()}])
    assert md.to_markdown('source') == ['Converted text']


def test_preserved_tags_round_trip_with_attributes_and_comments():
    source = '<div data-id="123">Hello<!--keep--></div>'
    md = converter([{'content': source}])
    def respond(messages):
        markers = re.findall(r'ET_TAG_[a-f0-9]+_\d+_END', str(messages[1]['content']))
        return markers[0] + '**Hello**' + markers[1] + markers[2]
    md.llm.raw_completion.side_effect = respond
    assert md.to_markdown('source', preserve_tags=True) == ['<div data-id="123">**Hello**<!--keep--></div>']
    assert md.document_loader.load.return_value[0]['content'] == source


def test_missing_tag_markers_raise_instead_of_silently_losing_tags():
    md = converter([{'content': '<div>Hello</div>'}])
    with pytest.raises(ValueError, match='page 1') as error:
        md.to_markdown('source', preserve_tags=True)
    assert 'preserve all source tags' in str(error.value.__cause__)


@pytest.mark.parametrize('pages', [[0], [True], [1, 1], [2]])
def test_bad_page_selection_does_not_call_model(pages):
    md = converter([{'content': 'one'}])
    with pytest.raises(ValueError):
        md.to_markdown('source', pages=pages)
    md.llm.raw_completion.assert_not_called()


def test_empty_selection_and_empty_page_keep_output_cardinality():
    md = converter([{'content': ''}], completion='')
    assert md.to_markdown('source', pages=[]) == []
    assert md.to_markdown('source') == ['']


def test_page_failure_is_not_converted_to_successful_markdown():
    md = converter([{'content': 'one'}])
    md.llm.raw_completion.side_effect = RuntimeError('provider failed')
    with pytest.raises(ValueError, match='page 1'):
        md.to_markdown('source')


def test_async_options_forwarded():
    md = converter([{'content': 'one', 'image': image_bytes()}])
    result = asyncio.run(md.to_markdown_async('source', include_images=True))
    assert 'data:image/png' in result[0]


def test_quoted_angle_brackets_in_tags_are_preserved():
    source = '<div title="1 > 0">Hello</div>'
    md = converter([{'content': source}])
    def respond(messages):
        markers = re.findall(r'ET_TAG_[a-f0-9]+_\d+_END', str(messages[1]['content']))
        return markers[0] + 'Hello' + markers[1]
    md.llm.raw_completion.side_effect = respond
    assert md.to_markdown('source', preserve_tags=True) == [source]


def test_structured_failure_does_not_return_string_instead_of_pagecontent():
    md = converter([{'content': 'one', 'image': image_bytes()}])
    md._process_page_with_llm = Mock(side_effect=RuntimeError('provider failure'))
    with pytest.raises(ValueError, match='Structured Markdown conversion failed'):
        md.to_markdown_structured('source')
    assert md.to_markdown_structured('source', pages=[]) == []
