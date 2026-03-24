"""Tests for MiniMax LLM provider integration.

Unit tests validate model detection, temperature clamping, and parameter
wiring without making any network calls.  Integration tests (marked with
``pytest.mark.integration``) require a valid ``MINIMAX_API_KEY`` environment
variable and perform real API calls.
"""

import os
import sys
from unittest.mock import patch

import pytest

# Import utility helpers directly (no heavy transitive deps)
from extract_thinker.utils import (
    is_minimax_model,
    resolve_minimax_model,
    MINIMAX_API_BASE,
)
from extract_thinker.global_models import get_minimax_model, get_minimax_highspeed_model

# LLM import may fail in some environments due to unrelated transitive deps.
# Guard it so the pure-utility tests still run.
_LLM_AVAILABLE = True
try:
    from extract_thinker.llm import LLM
except Exception:
    _LLM_AVAILABLE = False

needs_llm = pytest.mark.skipif(not _LLM_AVAILABLE, reason="LLM class not importable (env dep issue)")


# ---------------------------------------------------------------------------
# Unit tests – no network calls
# ---------------------------------------------------------------------------


class TestIsMiniMaxModel:
    """Test the is_minimax_model() helper."""

    def test_minimax_prefix(self):
        assert is_minimax_model("minimax/MiniMax-M2.7") is True

    def test_minimax_prefix_case_insensitive(self):
        assert is_minimax_model("MiniMax/MiniMax-M2.5") is True

    def test_openai_prefix_with_minimax_model(self):
        assert is_minimax_model("openai/MiniMax-M2.7") is True

    def test_bare_minimax_model_name(self):
        assert is_minimax_model("MiniMax-M2.7-highspeed") is True

    def test_non_minimax_model(self):
        assert is_minimax_model("gpt-4o") is False

    def test_non_minimax_provider(self):
        assert is_minimax_model("anthropic/claude-3") is False

    def test_ollama_model(self):
        assert is_minimax_model("ollama/phi4") is False


class TestResolveMiniMaxModel:
    """Test the resolve_minimax_model() rewriter."""

    def test_minimax_prefix_to_openai(self):
        assert resolve_minimax_model("minimax/MiniMax-M2.7") == "openai/MiniMax-M2.7"

    def test_already_openai_prefix(self):
        assert resolve_minimax_model("openai/MiniMax-M2.7") == "openai/MiniMax-M2.7"

    def test_bare_model_name(self):
        assert resolve_minimax_model("MiniMax-M2.7") == "openai/MiniMax-M2.7"

    def test_highspeed_variant(self):
        assert resolve_minimax_model("minimax/MiniMax-M2.7-highspeed") == "openai/MiniMax-M2.7-highspeed"

    def test_m25_variant(self):
        assert resolve_minimax_model("minimax/MiniMax-M2.5") == "openai/MiniMax-M2.5"


class TestGlobalModels:
    """Test global model helper functions."""

    def test_get_minimax_model(self):
        model = get_minimax_model()
        assert model == "minimax/MiniMax-M2.7"
        assert is_minimax_model(model) is True

    def test_get_minimax_highspeed_model(self):
        model = get_minimax_highspeed_model()
        assert model == "minimax/MiniMax-M2.7-highspeed"
        assert is_minimax_model(model) is True


@needs_llm
class TestLLMInit:
    """Test LLM constructor auto-detection for MiniMax."""

    def test_minimax_auto_detection(self):
        llm = LLM("minimax/MiniMax-M2.7")
        assert llm._is_minimax is True
        assert llm.model == "openai/MiniMax-M2.7"
        assert llm.api_base == MINIMAX_API_BASE

    def test_minimax_api_key_from_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}):
            llm = LLM("minimax/MiniMax-M2.7")
            assert llm.api_key == "test-key-123"

    def test_minimax_explicit_api_key(self):
        llm = LLM("minimax/MiniMax-M2.7", api_key="explicit-key")
        assert llm.api_key == "explicit-key"

    def test_minimax_explicit_api_base(self):
        llm = LLM("minimax/MiniMax-M2.7", api_base="https://custom.api/v1")
        assert llm.api_base == "https://custom.api/v1"

    def test_non_minimax_no_auto_config(self):
        llm = LLM("gpt-4o")
        assert llm._is_minimax is False
        assert llm.api_base is None
        assert llm.api_key is None
        assert llm.model == "gpt-4o"

    def test_non_minimax_with_explicit_api_base(self):
        llm = LLM("gpt-4o", api_base="https://custom.openai/v1")
        assert llm.api_base == "https://custom.openai/v1"

    def test_highspeed_model_detection(self):
        llm = LLM("minimax/MiniMax-M2.7-highspeed")
        assert llm._is_minimax is True
        assert llm.model == "openai/MiniMax-M2.7-highspeed"


@needs_llm
class TestTemperatureClamping:
    """Test MiniMax temperature clamping."""

    def test_default_temperature(self):
        llm = LLM("minimax/MiniMax-M2.7")
        assert llm._effective_temperature() == 0

    def test_temperature_in_range(self):
        llm = LLM("minimax/MiniMax-M2.7")
        llm.set_temperature(0.7)
        assert llm._effective_temperature() == 0.7

    def test_temperature_clamped_high(self):
        llm = LLM("minimax/MiniMax-M2.7")
        llm.set_temperature(1.5)
        assert llm._effective_temperature() == 1.0

    def test_temperature_clamped_low(self):
        llm = LLM("minimax/MiniMax-M2.7")
        llm.set_temperature(-0.5)
        assert llm._effective_temperature() == 0.0

    def test_non_minimax_no_clamping(self):
        llm = LLM("gpt-4o")
        llm.set_temperature(1.5)
        assert llm._effective_temperature() == 1.5


@needs_llm
class TestExtraParams:
    """Test _get_extra_params() for MiniMax."""

    def test_minimax_extra_params(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key-abc"}):
            llm = LLM("minimax/MiniMax-M2.7")
            params = llm._get_extra_params()
            assert params["api_base"] == MINIMAX_API_BASE
            assert params["api_key"] == "key-abc"

    def test_non_minimax_empty_params(self):
        llm = LLM("gpt-4o")
        params = llm._get_extra_params()
        assert params == {}


@needs_llm
class TestLLMDynamic:
    """Test MiniMax with dynamic JSON parsing mode."""

    def test_minimax_dynamic_mode_enabled(self):
        llm = LLM("minimax/MiniMax-M2.7")
        llm.set_dynamic(True)
        assert llm.is_dynamic is True
        assert llm._is_minimax is True


# ---------------------------------------------------------------------------
# Integration tests – require MINIMAX_API_KEY
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field
from typing import Optional


class InvoiceFields(BaseModel):
    """Invoice-like contract for integration tests."""
    invoice_number: str = Field(description="The invoice number")
    invoice_date: str = Field(description="The invoice date")
    total_amount: Optional[float] = Field(default=None, description="Total amount")


@pytest.fixture
def minimax_api_key():
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        pytest.skip("MINIMAX_API_KEY not set")
    return key


@pytest.mark.integration
@needs_llm
class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_raw_completion(self, minimax_api_key):
        """Test raw text completion with MiniMax."""
        llm = LLM("minimax/MiniMax-M2.7")
        messages = [
            {"role": "user", "content": "Reply with exactly: Hello World"}
        ]
        result = llm.raw_completion(messages)
        assert result is not None
        assert len(result) > 0
        assert "hello" in result.lower()

    def test_extraction_with_extractor(self, minimax_api_key):
        """Test structured extraction via Extractor with MiniMax."""
        from extract_thinker import Extractor, DocumentLoaderPyPdf

        cwd = os.getcwd()
        test_file = os.path.join(cwd, "tests", "files", "invoice.pdf")
        if not os.path.exists(test_file):
            pytest.skip("Test invoice.pdf not found")

        extractor = Extractor()
        extractor.load_document_loader(DocumentLoaderPyPdf())
        extractor.load_llm("minimax/MiniMax-M2.7")

        result = extractor.extract(test_file, InvoiceFields)
        assert result is not None
        assert isinstance(result, InvoiceFields)
        assert result.invoice_number is not None
        assert len(result.invoice_number) > 0

    def test_highspeed_model(self, minimax_api_key):
        """Test the highspeed variant works."""
        llm = LLM("minimax/MiniMax-M2.7-highspeed")
        messages = [
            {"role": "user", "content": "What is 2 + 2? Reply with just the number."}
        ]
        result = llm.raw_completion(messages)
        assert result is not None
        assert "4" in result
