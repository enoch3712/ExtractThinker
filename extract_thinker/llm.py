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
    THINKING_BUDGET_TOKENS = 8000
    DEFAULT_PAGE_TOKENS = 1500  # Each page has this many tokens (text + image)
    DEFAULT_THINKING_RATIO = 1/3  # Thinking budget as a fraction of content tokens
    MAX_TOKEN_LIMIT = 120000  # Maximum token limit (for Claude 3.7 Sonnet)
    MAX_THINKING_BUDGET = 64000  # Maximum thinking budget
    MIN_THINKING_BUDGET = 1200  # Minimum thinking budget

    # Model-specific token limits
    MODEL_TOKEN_LIMITS = {
        "gpt-4o": 12000,  # Reasonable middle ground for GPT-4o
    }

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
        self.is_thinking = False  # Initialize is_thinking flag
        self.page_count = None  # Initialize page count
        self.thinking_budget = self.THINKING_BUDGET_TOKENS  # Default thinking budget

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

    def _get_model_max_tokens(self) -> int:
        """Get the maximum tokens allowed for the current model."""
        # Only apply special limit for GPT-4o
        if self.model == "gpt-4o":
            return self.MODEL_TOKEN_LIMITS["gpt-4o"]

        # Default to the general MAX_TOKEN_LIMIT for all other models
        return self.MAX_TOKEN_LIMIT

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

    def set_thinking(self, is_thinking: bool) -> None:
        """Set whether the LLM should handle thinking.

        Args:
            is_thinking (bool): Whether to enable thinking
        """
        self.is_thinking = is_thinking
        self.temperature = 1

    def set_dynamic(self, is_dynamic: bool) -> None:
        """Set whether the LLM should handle dynamic content.

        When dynamic is True, the LLM will attempt to parse and validate JSON responses.
        This is useful for handling structured outputs like masking mappings.

        Args:
            is_dynamic (bool): Whether to enable dynamic content handling
        """
        self.is_dynamic = is_dynamic

    def set_page_count(self, page_count: int) -> None:
        """Set the page count to calculate token limits for thinking.

        Each page is assumed to have DEFAULT_PAGE_TOKENS tokens (text + image).
        Thinking budget is calculated as DEFAULT_THINKING_RATIO of the content tokens.

        Args:
            page_count (int): Number of pages in the document
        """
        if page_count <= 0:
            raise ValueError("Page count must be a positive integer")

        self.page_count = page_count

        # Calculate content tokens
        content_tokens = min(page_count * self.DEFAULT_PAGE_TOKENS, self.MAX_TOKEN_LIMIT)

        # Calculate thinking budget (1/3 of content tokens)
        thinking_tokens = int(page_count * self.DEFAULT_PAGE_TOKENS * self.DEFAULT_THINKING_RATIO)

        # Apply min/max constraints
        thinking_tokens = max(thinking_tokens, self.MIN_THINKING_BUDGET)
        thinking_tokens = min(thinking_tokens, self.MAX_THINKING_BUDGET)

        # Update token limit and thinking budget
        self.token_limit = content_tokens
        self.thinking_budget = thinking_tokens

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

        # Use router or direct call based on thinking state
        if self.router:
            response = self._request_with_router(working_messages, request_model)
        else:
            response = self._request_direct(working_messages, request_model)

        # If response_model is provided, return the response directly
        if self.is_dynamic == False:
            return response

        # Otherwise get content and handle dynamic parsing if enabled
        content = response.choices[0].message.content
        if self.is_dynamic:
            return extract_thinking_json(content, response_model)

        return content

    def _request_with_router(self, messages: List[Dict[str, str]], response_model: Optional[str]) -> Any:
        """Handle request using router with or without thinking parameter"""
        if self.is_thinking:
            # Add thinking parameter for supported models
            thinking_param = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }
            try:
                return self.router.completion(
                    model=self.model,
                    messages=messages,
                    response_model=response_model,
                    temperature=self.temperature,
                    timeout=self.TIMEOUT,
                    thinking=thinking_param,
                )
            except Exception as e:
                # If thinking parameter causes an error, try without it
                if "property 'thinking' is unsupported" in str(e):
                    print(f"Warning: Model {self.model} doesn't support thinking parameter, proceeding without it.")
                    return self.router.completion(
                        model=self.model,
                        messages=messages,
                        response_model=response_model,
                        temperature=self.temperature,
                        timeout=self.TIMEOUT,
                    )
                else:
                    raise e
        else:
            # Normal request without thinking parameter
            return self.router.completion(
                model=self.model,
                messages=messages,
                response_model=response_model,
                temperature=self.temperature,
                timeout=self.TIMEOUT,
            )

    def _request_direct(self, messages: List[Dict[str, str]], response_model: Optional[str]) -> Any:
        """Handle direct request with or without thinking parameter"""
        base_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_model": response_model,
            "max_retries": 1,
            "max_tokens": self._get_model_max_tokens(),  # <- capped max tokens here
            "timeout": self.TIMEOUT,
        }

        if self.is_thinking:
            # Try with thinking parameter
            thinking_param = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }
            try:
                return self.client.chat.completions.create(
                    **base_params,
                    thinking=thinking_param,
                )
            except Exception as e:
                # If thinking parameter causes an error, try without it
                if "property 'thinking' is unsupported" in str(e):
                    print(f"Warning: Model {self.model} doesn't support thinking parameter, proceeding without it.")
                    return self.client.chat.completions.create(**base_params)
                else:
                    raise e
        else:
            # Normal request without thinking parameter
            return self.client.chat.completions.create(**base_params)

    def raw_completion(self, messages: List[Dict[str, str]]) -> str:
        """Make raw completion request without response model."""
        max_tokens = self._get_model_max_tokens()  # <- capped max tokens here

        if self.router:
            if self.is_thinking:
                # Add thinking parameter for supported models
                thinking_param = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget
                }
                try:
                    raw_response = self.router.completion(
                        model=self.model,
                        messages=messages,
                        thinking=thinking_param,
                    )
                except Exception as e:
                    # If thinking parameter causes an error, try without it
                    if "property 'thinking' is unsupported" in str(e):
                        print(f"Warning: Model {self.model} doesn't support thinking parameter, proceeding without it.")
                        raw_response = self.router.completion(
                            model=self.model,
                            messages=messages,
                        )
                    else:
                        raise e
            else:
                raw_response = self.router.completion(
                    model=self.model,
                    messages=messages,
                )
        else:
            if self.is_thinking:
                # Add thinking parameter for supported models
                thinking_param = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget
                }
                try:
                    raw_response = litellm.completion(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        thinking=thinking_param,
                    )
                except Exception as e:
                    # If thinking parameter causes an error, try without it
                    if "property 'thinking' is unsupported" in str(e):
                        print(f"Warning: Model {self.model} doesn't support thinking parameter, proceeding without it.")
                        raw_response = litellm.completion(
                            model=self.model,
                            messages=messages,
                            max_tokens=max_tokens,
                        )
                    else:
                        raise e
            else:
                raw_response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
        return raw_response.choices[0].message.content

    def set_timeout(self, timeout_ms: int) -> None:
        """Set the timeout value for LLM requests in milliseconds."""
        self.TIMEOUT = timeout_ms
