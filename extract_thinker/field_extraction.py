"""Field-level model policies and bounded parallel extraction."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Optional, Union

from pydantic import BaseModel, create_model
from pydantic.functional_validators import AfterValidator, BeforeValidator, PlainValidator, WrapValidator
from extract_thinker.llm import LLM
from extract_thinker.models.completion_strategy import CompletionStrategy


@dataclass(frozen=True)
class FieldExtraction:
    """Use in Annotated fields to select a model, group, vision and instructions."""
    model: Optional[Union[str, LLM]] = None
    group: Optional[str] = None
    vision: Optional[bool] = None
    instructions: Optional[str] = None

    def __deepcopy__(self, memo):
        # Policies hold provider clients; copying a schema must not clone their
        # locks/connections. Each worker separately copies its mutable LLM state.
        return self

    def __post_init__(self):
        if self.model is not None and not isinstance(self.model, LLM) and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError('model must be a nonempty name or LLM instance')
        if self.instructions is not None and not isinstance(self.instructions, str):
            raise ValueError('instructions must be a string')
        if self.group is not None and (not isinstance(self.group, str) or not self.group):
            raise ValueError('group must be a nonempty string')
        if self.vision is not None and not isinstance(self.vision, bool):
            raise ValueError('vision must be True, False or None')


def has_field_extraction(response_model):
    return isinstance(response_model, type) and issubclass(response_model, BaseModel) and any(
        isinstance(item, FieldExtraction)
        for field in response_model.model_fields.values() for item in field.metadata
    )


def _groups(response_model):
    groups = {}
    for name, field in response_model.model_fields.items():
        policies = [item for item in field.metadata if isinstance(item, FieldExtraction)]
        if len(policies) > 1:
            raise ValueError(f'Field {name} has more than one FieldExtraction policy')
        policy = policies[0] if policies else FieldExtraction()
        key = ('group', policy.group) if policy.group else ('field', name)
        if key in groups and groups[key][0] != policy:
            raise ValueError(f'Fields in group {policy.group!r} must use the same extraction policy')
        groups.setdefault(key, (policy, []))[1].append(name)
    return list(groups.values())


def _partial_contract(response_model, names, index):
    fields = {}
    deferred = (FieldExtraction, BeforeValidator, AfterValidator, WrapValidator, PlainValidator)
    for name in names:
        source = response_model.model_fields[name]
        field = deepcopy(source)
        field.metadata = [item for item in field.metadata if not isinstance(item, deferred)]
        annotation = source.annotation
        if not source.is_required():
            # Apply defaults/factories only when validating the final contract.
            annotation = Optional[annotation]
            field.default = None
            field.default_factory = None
        fields[name] = (annotation, field)
    return create_model(f'{response_model.__name__}Part{index}', __config__=response_model.model_config, **fields)


def _load_pages(extractor, source, vision):
    if isinstance(source, list) and all(isinstance(page, dict) for page in source):
        return source
    if isinstance(source, dict):
        return [source]
    sources = source if isinstance(source, list) else [source]
    pages = []
    for item in sources:
        loader = extractor.get_document_loader(item)
        if loader is None:
            raise ValueError('No suitable document loader found for parallel extraction')
        previous = loader.vision_mode
        try:
            loader.set_vision_mode(vision)
            loaded = loader.load(item)
        finally:
            loader.set_vision_mode(previous)
        if isinstance(loaded, dict):
            loaded = [loaded]
        if not isinstance(loaded, list) or not all(isinstance(page, dict) for page in loaded):
            raise ValueError('Parallel extraction requires page dictionaries')
        pages.extend(loaded)
    return pages


def extract_fields(extractor, source, response_model, vision=False, content=None,
                   completion_strategy=CompletionStrategy.FORBIDDEN, max_workers=4):
    """Extract each field/group independently and validate the complete result."""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError('max_workers must be a positive integer')
    if not isinstance(response_model, type) or not issubclass(response_model, BaseModel):
        raise ValueError('response_model must be a Pydantic model class')
    groups = _groups(response_model)
    if not groups:
        return response_model.model_validate({})
    needs_images = any(policy.vision if policy.vision is not None else vision for policy, _ in groups)
    setup = copy(extractor)
    if setup.document_loader is None and (isinstance(source, dict) or
            isinstance(source, list) and all(isinstance(page, dict) for page in source)):
        from extract_thinker.document_loader.document_loader_data import DocumentLoaderData
        setup.document_loader = DocumentLoaderData()
    setup._validate_dependencies(response_model, needs_images)
    if needs_images and setup.document_loader is None:
        setup._handle_vision_mode(source)
    pages = _load_pages(setup, source, needs_images)
    if not pages:
        raise ValueError('No pages available for field extraction')
    extractor.llm.last_completion = None

    def run(index, policy, names):
        worker = copy(extractor)
        if isinstance(policy.model, str):
            worker.llm = LLM(policy.model)
        else:
            worker.llm = copy(policy.model if policy.model is not None else extractor.llm)
        worker.llm.last_completion = None
        worker.llm.set_page_count(max(1, len(pages)))
        worker.llm_interceptors = list(extractor.llm_interceptors)
        worker.extra_content = '\n\n'.join(part for part in (content, policy.instructions) if part) or None
        worker.completion_strategy = completion_strategy
        worker.allow_vision = policy.vision if policy.vision is not None else vision
        contract = _partial_contract(response_model, names, index)
        if completion_strategy == CompletionStrategy.FORBIDDEN:
            universal = worker._map_to_universal_format(pages, worker.allow_vision)
            result = worker._extract(universal, contract, worker.allow_vision)
        else:
            result = worker.extract_with_strategy(pages, contract, worker.allow_vision, completion_strategy)
        if not isinstance(result, contract):
            result = contract.model_validate(result)
        return {name: getattr(result, name) for name in names if name in result.model_fields_set}

    combined = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(groups))) as executor:
        futures = {executor.submit(run, index, policy, names): names
                   for index, (policy, names) in enumerate(groups)}
        for future in as_completed(futures):
            try:
                combined.update(future.result())
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise ValueError(f'Field extraction failed for {futures[future]}: {exc}') from exc
    # Root/model/field validators see the full assembled contract only here.
    return response_model.model_validate(combined, by_name=True)
