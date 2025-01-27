from typing import List, Dict, Any, Optional
import instructor
import litellm
from litellm import Router
from extract_thinker.utils import add_classification_structure, extract_thinking_json

# Add these constants at the top of the file, after the imports
DYNAMIC_PROMPT_TEMPLATE = """Please provide your thinking process within <think> tags, followed by your JSON output.

JSON structure:
{prompt}

OUTPUT example:
<think>
Your step-by-step reasoning and analysis goes here...
</think>

##JSON OUTPUT
{{
    ...
}}
"""

class LLM:
    TEMPERATURE = 0  # Always zero for deterministic outputs (IDP)
    TIMEOUT = 3000  # Timeout in milliseconds

    def __init__(self,
                 model: str,
                 token_limit: int = None):
        self.client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        self.model = model
        self.router = None
        self.token_limit = token_limit
        self.is_dynamic = False
    def load_router(self, router: Router) -> None:
        self.router = router

    def set_dynamic(self, is_dynamic: bool) -> None:
        """Set whether the LLM should handle dynamic content.
        
        When dynamic is True, the LLM will attempt to parse and validate JSON responses.
        This is useful for handling structured outputs like masking mappings.
        
        Args:
            is_dynamic (bool): Whether to enable dynamic content handling
        """
        self.is_dynamic = is_dynamic

    def request(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[str] = None
    ) -> Any:
        # Uncomment the following lines if you need to calculate max_tokens
        # contents = map(lambda message: message['content'], messages)
        # all_contents = ' '.join(contents)
        # max_tokens = num_tokens_from_string(all_contents)

        # if is sync, response model is None if dynamic true and used for dynamic parsing after llm request
        request_model = None if self.is_dynamic else response_model

        # Add model structure and prompt engineering if dynamic parsing is enabled
        working_messages = messages.copy()
        if self.is_dynamic and response_model:
            structure = add_classification_structure(response_model)
            prompt = DYNAMIC_PROMPT_TEMPLATE.format(prompt=structure)
            working_messages.append({
                "role": "system",
                "content": prompt
            })

        if self.router:
            response = self.router.completion(
                model=self.model,
                messages=working_messages,
                response_model=request_model,
                temperature=self.TEMPERATURE,
                timeout=self.TIMEOUT,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=working_messages,
                temperature=self.TEMPERATURE,
                response_model=request_model,
                max_retries=1,
                max_tokens=self.token_limit,
                timeout=self.TIMEOUT,
            )

        # If response_model is provided, return the response directly
        if self.is_dynamic == False:
            return response

        # Otherwise get content and handle dynamic parsing if enabled
        content = response.choices[0].message.content
        if self.is_dynamic:
            return extract_thinking_json(content, response_model)
            
        return content

    def raw_completion(self, messages: List[Dict[str, str]]) -> str:
        """Make raw completion request without response model."""
        if self.router:
            raw_response = self.router.completion(
                model=self.model,
                messages=messages
            )
        else:
            raw_response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=self.token_limit
            )
        return raw_response.choices[0].message.content

    def set_timeout(self, timeout_ms: int) -> None:
        """Set the timeout value for LLM requests in milliseconds."""
        self.TIMEOUT = timeout_ms