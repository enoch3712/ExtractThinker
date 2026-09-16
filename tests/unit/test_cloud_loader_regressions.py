"""Offline regressions for issues #247, #347 and #150."""
from io import BytesIO
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from extract_thinker import (
    AzureConfig, DocumentLoaderAzureForm, DocumentLoaderAWSTextract, TextractConfig,
)


@pytest.mark.parametrize('use_config', [True, False])
def test_temporary_aws_credentials_reach_sdk(monkeypatch, use_config):
    client = Mock()
    monkeypatch.setattr('boto3.client', client)
    kwargs = dict(aws_access_key_id='test-key', aws_secret_access_key='test-secret',
                  aws_session_token='test-token', region_name='us-east-1')
    loader = (DocumentLoaderAWSTextract(TextractConfig(**kwargs)) if use_config
              else DocumentLoaderAWSTextract(**kwargs))
    client.assert_called_once_with('textract', **kwargs)
    assert loader.textract_client is client.return_value


def test_aws_default_credential_chain(monkeypatch):
    client = Mock()
    monkeypatch.setattr('boto3.client', client)
    DocumentLoaderAWSTextract(region_name='us-east-1')
    client.assert_called_once_with('textract', region_name='us-east-1')


def test_aws_injected_client_takes_precedence(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr('boto3.client', constructor)
    client = Mock()
    assert DocumentLoaderAWSTextract.from_client(client).textract_client is client
    constructor.assert_not_called()


@pytest.mark.parametrize('kwargs', [dict(aws_access_key_id='key'), dict(aws_secret_access_key='secret')])
def test_incomplete_explicit_credentials_fail_before_sdk(monkeypatch, kwargs):
    client = Mock()
    monkeypatch.setattr('boto3.client', client)
    with pytest.raises(ValueError, match='both AWS'):
        DocumentLoaderAWSTextract(**kwargs)
    client.assert_not_called()


@pytest.mark.parametrize('construction', ['config', 'legacy', 'factory'])
def test_azure_api_version_reaches_sdk(monkeypatch, construction):
    client = Mock()
    monkeypatch.setattr('azure.ai.formrecognizer.DocumentAnalysisClient', client)
    kwargs = dict(subscription_key='test-key', endpoint='https://example.cognitiveservices.azure.com',
                  api_version='2022-08-31')
    if construction == 'config':
        DocumentLoaderAzureForm(AzureConfig(**kwargs))
    elif construction == 'factory':
        DocumentLoaderAzureForm.from_credentials(**kwargs)
    else:
        DocumentLoaderAzureForm(**kwargs)
    assert client.call_args.kwargs['api_version'] == '2022-08-31'


def test_azure_uses_sdk_default_when_version_unset(monkeypatch):
    client = Mock()
    monkeypatch.setattr('azure.ai.formrecognizer.DocumentAnalysisClient', client)
    DocumentLoaderAzureForm('key', 'https://example.cognitiveservices.azure.com')
    assert 'api_version' not in client.call_args.kwargs


@pytest.fixture
def azure(monkeypatch):
    monkeypatch.setattr('azure.ai.formrecognizer.DocumentAnalysisClient', Mock())
    return DocumentLoaderAzureForm('key', 'https://example.cognitiveservices.azure.com')


def table(page, cells, rows=1, columns=3):
    return NS(row_count=rows, column_count=columns,
              bounding_regions=[NS(page_number=page)],
              cells=[NS(row_index=r, column_index=c, content=text) for r, c, text in cells])


def test_azure_preserves_missing_cells_order_and_multiple_tables(azure):
    tables = [table(1, [(0, 2, 'last'), (0, 0, 'first')]),
              table(1, [(0, 1, 'second table')]),
              table(2, [(0, 0, None)])]
    assert azure.build_tables(tables) == {
        1: [[['first', '', 'last']], [['', 'second table', '']]],
        2: [[['', '', '']]],
    }


def test_azure_removes_cell_lines_without_mutating_input(azure):
    paragraphs = ['A', 'A', 'Outside', 'B']
    assert azure.remove_lines_present_in_tables(paragraphs, [[['A', '', 'B']]]) == ['Outside']
    assert paragraphs == ['A', 'A', 'Outside', 'B']


def test_azure_table_only_content_excludes_prose(azure, monkeypatch):
    azure.config.content_mode = 'tables'
    azure.client.begin_analyze_document.return_value.result.return_value = NS(
        pages=[NS(page_number=1, lines=[NS(content='unwanted prose')])],
        tables=[table(1, [(0, 0, 'A'), (0, 2, 'B')])],
        key_value_pairs=None,
    )
    monkeypatch.setattr(azure, 'can_handle', lambda _: True)
    pages = azure.load(BytesIO(b'fake document'))
    assert pages[0]['content'] == '[[["A", "", "B"]]]'
    assert pages[0]['tables'] == [[['A', '', 'B']]]


def test_azure_non_table_mode_handles_missing_optional_values(azure, monkeypatch):
    azure.client.begin_analyze_document.return_value.result.return_value = NS(
        pages=[NS(page_number=1, lines=None)], tables=None, key_value_pairs=None, languages=None,
    )
    monkeypatch.setattr(azure, 'can_handle', lambda _: True)
    assert azure.load(BytesIO(b'fake document')) == [{'content': '', 'tables': [], 'forms': {}, 'languages': []}]


def test_azure_rejects_invalid_content_mode():
    with pytest.raises(ValueError, match='content_mode'):
        AzureConfig('key', 'endpoint', content_mode='typo')
