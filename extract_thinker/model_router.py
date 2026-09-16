"""Configurable model selection based on measurable request complexity."""
import math
from dataclasses import dataclass
from threading import RLock
from typing import Callable, List, Optional, Union

from extract_thinker.llm import LLM


@dataclass(frozen=True)
class RequestComplexity:
    text_characters: int
    image_count: int
    field_count: int
    schema_depth: int
    page_count: int

    @property
    def score(self):
        return (self.text_characters / 4000 + self.image_count * 4
                + self.field_count / 4 + self.schema_depth
                + max(0, self.page_count - 1) / 2)


@dataclass(frozen=True)
class ModelRoute:
    model: Union[str, LLM]
    max_score: Optional[float] = None
    supports_vision: bool = False


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    score: float
    complexity: RequestComplexity


def _schema_stats(schema):
    definitions = schema.get('$defs', {})
    def visit(node, seen):
        if not isinstance(node, dict):
            return 0, 0
        reference = node.get('$ref')
        if reference:
            if reference in seen:
                return 0, 0
            return visit(definitions.get(reference.rsplit('/', 1)[-1], {}), seen | {reference})
        properties = node.get('properties', {})
        children = list(properties.values())
        if isinstance(node.get('items'), dict):
            children.append(node['items'])
        for keyword in ('anyOf', 'allOf', 'oneOf'):
            children.extend(node.get(keyword, []))
        stats = [visit(child, seen) for child in children]
        return (len(properties) + sum(count for count, _ in stats),
                (1 if properties else 0) + max((depth for _, depth in stats), default=0))
    return visit(schema, set())


def _message_stats(messages):
    characters = images = 0
    for message in messages:
        content = message.get('content', '')
        if isinstance(content, str):
            characters += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') in ('image_url', 'input_image', 'image'):
                    images += 1
                elif isinstance(block, dict):
                    characters += len(str(block.get('text', '')))
                elif isinstance(block, str):
                    characters += len(block)
    return characters, images


class ComplexityRouter(LLM):
    """Select the first capable route whose score threshold fits the request.

    The score is a configurable heuristic, not a quality or cost prediction.
    A single router serializes calls to keep mutable provider state isolated.
    Provider batch requests bypass routing and are therefore unsupported.
    """
    supports_batch = False

    def __init__(self, routes: List[ModelRoute], scorer: Optional[Callable[[RequestComplexity], float]] = None):
        if not routes:
            raise ValueError('At least one model route is required')
        previous = -1
        for index, route in enumerate(routes):
            if not isinstance(route.model, LLM) and (not isinstance(route.model, str) or not route.model.strip()):
                raise ValueError('Each route requires a model name or LLM instance')
            if route.max_score is None:
                if index != len(routes) - 1:
                    raise ValueError('An unbounded route must be last')
            elif isinstance(route.max_score, bool) or not isinstance(route.max_score, (int, float)) or not math.isfinite(route.max_score) or route.max_score < 0 or route.max_score <= previous:
                raise ValueError('Route thresholds must be finite, nonnegative and strictly increasing')
            else:
                previous = route.max_score
        self._routes = [(route, route.model if isinstance(route.model, LLM) else LLM(route.model)) for route in routes]
        super().__init__(self._routes[0][1].model)
        self._lock = RLock()
        self.scorer = scorer or (lambda complexity: complexity.score)
        self.last_decision = None

    def _choose(self, messages, response_model=None):
        text, images = _message_stats(messages)
        fields, depth = _schema_stats(response_model.model_json_schema()) if response_model is not None else (0, 0)
        complexity = RequestComplexity(text, images, fields, depth, self.page_count or 1)
        score = self.scorer(complexity)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
            raise ValueError('Complexity scorer must return a finite nonnegative number')
        for route, llm in self._routes:
            if images and not route.supports_vision:
                continue
            if route.max_score is None or score <= route.max_score:
                self.last_decision = RoutingDecision(llm.model, score, complexity)
                self.model = llm.model
                llm.set_page_count(self.page_count or 1)
                return llm
        raise ValueError('No configured route supports this request complexity and image input')

    def request(self, messages, response_model=None):
        with self._lock:
            self.last_completion = None
            self.last_decision = None
            llm = self._choose(messages, response_model)
            result = llm.request(messages, response_model)
            self.last_completion = llm.last_completion
            return result

    def raw_completion(self, messages):
        with self._lock:
            self.last_completion = None
            self.last_decision = None
            llm = self._choose(messages)
            result = llm.raw_completion(messages)
            self.last_completion = llm.last_completion
            return result

    def set_temperature(self, temperature):
        for _, llm in self._routes:
            llm.set_temperature(temperature)
        super().set_temperature(temperature)

    def set_thinking(self, enabled):
        for _, llm in self._routes:
            llm.set_thinking(enabled)
        super().set_thinking(enabled)

    def set_dynamic(self, enabled):
        for _, llm in self._routes:
            llm.set_dynamic(enabled)
        super().set_dynamic(enabled)

    def set_timeout(self, timeout_ms):
        for _, llm in self._routes:
            llm.set_timeout(timeout_ms)
        super().set_timeout(timeout_ms)

    def load_router(self, router):
        raise ValueError('Configure fallback routers on individual route LLMs')
