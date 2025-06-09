def get_lite_model():
    """Return the lite model for cost efficiency."""
    #return "vertex_ai/gemini-2.0-flash"
    return "gemini/gemini-2.5-flash-preview-05-20"

def get_big_model():
    """Return the big model for high performance."""
    #return "vertex_ai/gemini-2.0-flash"
    return "gemini/gemini-2.5-flash-preview-05-20"

def get_gemini_flash_model():
    """Return the Gemini 2.5 Flash model."""
    return "gemini/gemini-2.5-flash-preview-05-20"

def get_gpt_mini_model():
    """Return the GPT-4.1 Mini model."""
    return "gpt-4.1-mini"

def get_gpt_o4_model():
    """Return the GPT-4o model."""
    return "gpt-4o"