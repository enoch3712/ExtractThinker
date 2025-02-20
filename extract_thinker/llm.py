import asyncio
from typing import List, Dict, Any, Optional
import instructor
import litellm
from litellm import Router
from extract_thinker.llm_engine import LLMEngine
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
    TIMEOUT = 3000  # Timeout in milliseconds
    DEFAULT_TEMPERATURE = 0

    def __init__(
        self,
        model: str,
        token_limit: int = None,
        backend: LLMEngine = LLMEngine.DEFAULT
    ):
        """Initialize LLM with specified backend.
        
        Args:
            model: The model name (e.g. "gpt-4", "claude-3")
            token_limit: Optional maximum tokens
            backend: LLMBackend enum (default: LITELLM)
        """
        self.model = model
        self.token_limit = token_limit
        self.router = None
        self.is_dynamic = False
        self.backend = backend
        self.temperature = self.DEFAULT_TEMPERATURE

        if self.backend == LLMEngine.DEFAULT:
            self.client = instructor.from_litellm(
                litellm.completion,
                mode=instructor.Mode.MD_JSON
            )
            self.agent = None
        elif self.backend == LLMEngine.PYDANTIC_AI:
            self._check_pydantic_ai()
            from pydantic_ai import Agent
            from pydantic_ai.models import KnownModelName
            from typing import cast
            import asyncio
            
            self.client = None
            self.agent = Agent(
                cast(KnownModelName, self.model)
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @staticmethod
    def _check_pydantic_ai():
        """Check if pydantic-ai is installed."""
        try:
            import pydantic_ai
        except ImportError:
            raise ImportError(
                "Could not import pydantic-ai package. "
                "Please install it with `pip install pydantic-ai`."
            )

    @staticmethod
    def _get_pydantic_ai():
        """Lazy load pydantic-ai."""
        try:
            import pydantic_ai
            return pydantic_ai
        except ImportError:
            raise ImportError(
                "Could not import pydantic-ai package. "
                "Please install it with `pip install pydantic-ai`."
            )

    def load_router(self, router: Router) -> None:
        """Load a LiteLLM router for model fallbacks."""
        if self.backend != LLMEngine.DEFAULT:
            raise ValueError("Router is only supported with LITELLM backend")
        self.router = router

    def set_temperature(self, temperature: float) -> None:
        """Set the temperature for LLM requests.
        
        Args:
            temperature (float): Temperature value between 0 and 1
        """
        self.temperature = temperature

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
        # Handle Pydantic-AI backend differently
        if self.backend == LLMEngine.PYDANTIC_AI:
            # Combine messages into a single prompt
            combined_prompt = " ".join([m["content"] for m in messages])
            try:
                # Create event loop if it doesn't exist
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    self.agent.run(
                        combined_prompt, 
                        result_type=response_model if response_model else str
                    )
                )
                return result.data
            except Exception as e:
                raise ValueError(f"Failed to extract from source: {str(e)}")

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
                temperature=self.temperature,
                timeout=self.TIMEOUT,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=working_messages,
                temperature=self.temperature,
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