"""Uses real SDK job models with a fake service; never uploads a document."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

pytest.importorskip('adobe.pdfservices')
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
from extract_thinker import DocumentLoaderAdobePDF


def test_real_sdk_job_with_offline_service(monkeypatch):
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf import extract_pdf_params as params_module
    real_params = params_module.ExtractPDFParams
    captured = []
    def create_params(**kwargs):
        result = real_params(**kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(params_module, 'ExtractPDFParams', create_params)
    client = Mock()
    client.upload.return_value = CloudAsset('urn:aaid:AS:UE1:offline-test')
    client.submit.return_value = 'https://example.invalid/result'
    result_asset = CloudAsset('urn:aaid:AS:UE1:offline-result')
    client.get_job_result.return_value = SimpleNamespace(get_result=lambda: SimpleNamespace(get_resource=lambda: result_asset))
    client.get_content.return_value = SimpleNamespace(get_input_stream=lambda: b'archive bytes')
    loader = DocumentLoaderAdobePDF(client=client)
    assert loader._extract_archive(b'%PDF-test') == b'archive bytes'
    client.upload.assert_called_once()
    assert client.upload.call_args.kwargs['input_stream'] == b'%PDF-test'
    job = client.submit.call_args.args[0]
    assert isinstance(job, ExtractPDFJob)
    params = captured[0]
    assert params.get_table_structure_type() == TableStructureType.CSV
    assert ExtractElementType.TABLES.value in params.get_elements_to_extract()
    client.get_content.assert_called_once_with(result_asset)


def test_missing_credentials_error(monkeypatch):
    monkeypatch.delenv('PDF_SERVICES_CLIENT_ID', raising=False)
    monkeypatch.delenv('PDF_SERVICES_CLIENT_SECRET', raising=False)
    with pytest.raises(ValueError, match='Set both PDF_SERVICES_CLIENT_ID'):
        DocumentLoaderAdobePDF()
