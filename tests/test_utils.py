import sys
import pytest
from extract_thinker.utils import num_tokens_from_string, simple_token_counter

def test_token_counter():
    test_text = "Hello world! This is a test."
    
    # Test the counter
    token_count = num_tokens_from_string(test_text)
    assert token_count > 0
    
    # Test simple counter directly
    simple_count = simple_token_counter(test_text)
    assert simple_count > 0
    
    # If Python <3.13, verify tiktoken is being used
    if sys.version_info[:2] < (3, 13):
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            tiktoken_count = len(encoding.encode(test_text))
            assert token_count == tiktoken_count
        except ImportError:
            pytest.skip("tiktoken not installed")

def test_empty_string():
    assert num_tokens_from_string("") == 0
    assert simple_token_counter("") == 0

def test_special_characters():
    assert num_tokens_from_string("@#$%^&*") > 0
    assert simple_token_counter("@#$%^&*") > 0

def test_numbers():
    assert num_tokens_from_string("123 456 789") > 0
    assert simple_token_counter("123 456 789") > 0 