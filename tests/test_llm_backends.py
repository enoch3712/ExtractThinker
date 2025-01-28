import pytest
from extract_thinker import LLM, llm_engine

def test_litellm_backend():
    """Test default LiteLLM backend"""
    llm = LLM("gpt-4", backend=llm_engine.LITELLM)
    assert llm.backend == llm_engine.LITELLM
    assert llm.client is not None
    assert llm.agent is None

def test_pydanticai_backend():
    """Test PydanticAI backend if available"""
    try:
        import pydantic_ai
        llm = LLM("gpt-4", backend=llm_engine.PYDANTIC_AI)
        assert llm.backend == llm_engine.PYDANTIC_AI
        assert llm.client is None
        assert llm.agent is not None
    except ImportError:
        pytest.skip("pydantic-ai not installed")

def test_invalid_backend():
    """Test invalid backend type raises error"""
    with pytest.raises(TypeError):
        LLM("gpt-4", backend="invalid")  # Should be LLMBackend enum

def test_router_with_pydanticai():
    """Test router not supported with PydanticAI"""
    from litellm import Router
    router = Router(model_list=[{"model_name": "gpt-4"}])
    
    llm = LLM("gpt-4", backend=llm_engine.PYDANTIC_AI)
    with pytest.raises(ValueError, match="Router is only supported with LITELLM backend"):
        llm.load_router(router)