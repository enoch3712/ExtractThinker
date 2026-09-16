from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
from PIL import Image
import pytest
from extract_thinker import (
    DocumentLoaderEvents, VisionEventDetector, PageSignals, EventRule,
    DocumentEventType, Extractor, Contract, LLM,
)
from extract_thinker.document_loader.document_loader import DocumentLoader


class Loader(DocumentLoader):
    def __init__(self, pages):
        super().__init__()
        self.pages = pages
        self.modes = []
    def can_handle(self, source):
        return True
    def load(self, source):
        self.modes.append(self.vision_mode)
        return self.pages


def signals(**kwargs):
    return PageSignals(contains_handwriting=kwargs.get('handwriting', False),
                       contains_charts=kwargs.get('charts', False),
                       contains_images=kwargs.get('images', False), summary='Observed content')


def test_built_in_events_and_custom_rules_with_source_identity():
    pages = [{'content': 'Urgent invoice', 'page_number': 7}, {'content': 'ordinary'}]
    detector = SimpleNamespace(detect=Mock(side_effect=[signals(handwriting=True, charts=True, images=True), signals()]))
    global_events, custom_events = [], []
    rule = EventRule('urgent', lambda page, detected: 'Urgent' in page['content'],
                     lambda event, page: custom_events.append((event.rule_name, page['content'])))
    loader = DocumentLoaderEvents(Loader(pages), detector, [rule], lambda event, page: global_events.append(event.type))
    result = loader.load('source')
    assert global_events == [DocumentEventType.HANDWRITING, DocumentEventType.CHART, DocumentEventType.IMAGE, DocumentEventType.RULE]
    assert custom_events == [('urgent', 'Urgent invoice')]
    assert all(event.page_number == 7 for event in loader.last_events)
    assert result[1]['events'] == []
    assert result[0]['events'][0]['type'] == 'handwriting'
    assert 'events' not in pages[0]


def test_rule_only_processing_requires_no_model_or_images():
    callback = Mock()
    loader = DocumentLoaderEvents(Loader([{'content': 'Total: 120'}]),
        rules=[EventRule('has_total', lambda page, detected: detected is None and 'Total' in page['content'], callback)])
    loader.load('source')
    callback.assert_called_once()
    assert loader.last_events[0].rule_name == 'has_total'


def test_detector_failure_fires_no_partial_callbacks_and_restores_loader():
    original = Loader([{'content': 'first'}, {'content': 'second'}])
    detector = SimpleNamespace(requires_vision=True, detect=Mock(side_effect=[signals(charts=True), RuntimeError('failed')]))
    callback = Mock()
    loader = DocumentLoaderEvents(original, detector, on_event=callback)
    with pytest.raises(RuntimeError, match='failed'):
        loader.load('source')
    callback.assert_not_called()
    assert loader.last_events == []
    assert original.vision_mode is False
    assert original.modes == [True]


def test_callback_failure_propagates():
    callback = Mock(side_effect=RuntimeError('handler failed'))
    loader = DocumentLoaderEvents(Loader([{'content': 'text'}]),
        rules=[EventRule('all', lambda page, detected: True)], on_event=callback)
    with pytest.raises(RuntimeError, match='handler failed'):
        loader.load('source')


def test_vision_detector_uses_real_image_mime_and_structured_model():
    output = BytesIO()
    Image.new('RGB', (5, 5)).save(output, format='PNG')
    llm = LLM('test/vision')
    llm.request = Mock(return_value=signals(handwriting=True))
    result = VisionEventDetector(llm).detect({'image': output.getvalue()})
    assert result.contains_handwriting
    args = llm.request.call_args.args
    assert args[1] is PageSignals
    assert args[0][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
    assert llm.page_count is None


def test_event_metadata_reaches_extraction_without_sending_page_image():
    class Result(Contract):
        description: str
    output = BytesIO()
    Image.new('RGB', (5, 5)).save(output, format='PNG')
    detector_llm = LLM('test/vision')
    detector_llm.request = Mock(return_value=signals(handwriting=True))
    original = Loader([{'content': 'invoice', 'image': output.getvalue()}])
    loader = DocumentLoaderEvents(original, VisionEventDetector(detector_llm))
    extractor = Extractor(loader)
    extractor.llm = Mock()
    extractor.extract('source', Result)
    prompt = str(extractor.llm.request.call_args)
    assert 'handwriting' in prompt
    assert 'image_url' not in prompt
    assert original.modes == [True]


def test_no_page_images_is_an_explicit_error():
    with pytest.raises(ValueError, match='requires page images'):
        VisionEventDetector(LLM('test/vision')).detect({'content': 'plain'})


def test_duplicate_rules_are_rejected():
    rule = EventRule('same', lambda page, detected: True)
    with pytest.raises(ValueError, match='unique'):
        DocumentLoaderEvents(Loader([]), rules=[rule, rule])


def test_none_detector_result_does_not_mean_no_signals():
    detector = SimpleNamespace(detect=lambda page: None)
    with pytest.raises(ValueError):
        DocumentLoaderEvents(Loader([{'content': 'text'}]), detector).load('source')
