import asyncio
from typing import Any, List
from abc import ABC, abstractmethod

from extract_thinker.models.classification import Classification
from extract_thinker.models.doc_group import DocGroups, DocGroup
from extract_thinker.models.doc_groups2 import DocGroups2
from extract_thinker.models.eager_doc_group import EagerDocGroup


class Splitter(ABC):
    @abstractmethod
    def belongs_to_same_document(self, page1: Any, page2: Any, contract: str) -> DocGroups2:
        pass

    @abstractmethod
    def split_lazy_doc_group(self, lazy_doc_group: List[Any], classifications: List[Classification]) -> DocGroups:
        pass

    @abstractmethod
    def split_eager_doc_group(self, lazy_doc_group: List[Any], classifications: List[Classification]) -> DocGroups:
        pass

    def split_document_into_groups(self, document: List[Any]) -> List[List[Any]]:
        page_per_split = 2
        split = []
        if len(document) == 1:
            return [document]
        for i in range(0, len(document) - 1):
            group = document[i: i + page_per_split]
            split.append(group)
        return split

    async def process_split_groups(self, split: List[List[Any]], contract: str) -> List[DocGroups2]:
        # Create asynchronous tasks for processing each group
        tasks = [self.process_group(x, contract) for x in split]
        try:
            # Execute all tasks concurrently and wait for all to complete
            doc_groups = await asyncio.gather(*tasks)
            return doc_groups
        except Exception as e:
            # Handle possible exceptions that might occur during task execution
            print(f"An error occurred: {e}")
            raise

    async def process_group(self, group: List[Any], contract: str) -> DocGroups2:
        page2 = group[1] if len(group) > 1 else None
        return self.belongs_to_same_document(group[0], page2, contract)

    def _resolve_pair(self, response, classifications):
        from extract_thinker.models.split_classification import resolve_classification
        first = resolve_classification(classifications, response.classification_page1)
        second = resolve_classification(classifications, response.classification_page2)
        if response.belongs_to_same_document and response.classification_page1 != response.classification_page2:
            raise ValueError("Pages in the same document must have the same classification ID")
        return DocGroups2(
            reasoning=response.reasoning,
            belongs_to_same_document=response.belongs_to_same_document,
            classification_page1=first.name, classification_page2=second.name,
            classification_id_page1=response.classification_page1,
            classification_id_page2=response.classification_page2,
        )

    def _resolve_eager(self, response, classifications, page_count):
        from extract_thinker.models.split_classification import resolve_classification
        flattened = [page for group in response.groupOfDocuments for page in group.pages]
        if flattened != list(range(1, page_count + 1)) or any(not group.pages for group in response.groupOfDocuments):
            raise ValueError("Split groups must cover every page exactly once, in source order")
        return [EagerDocGroup(
            pages=group.pages,
            classification=resolve_classification(classifications, group.classification).name,
            classification_id=group.classification,
        ) for group in response.groupOfDocuments]

    def aggregate_doc_groups(self, doc_groups_tasks: List[DocGroups2]) -> DocGroups:
        """Combine adjacent comparisons without discarding classification conflicts."""
        result = DocGroups()
        if not doc_groups_tasks:
            return result
        first = doc_groups_tasks[0]
        current = DocGroup([1], first.classification_page1, first.classification_id_page1)
        previous_name, previous_id = first.classification_page1, first.classification_id_page1
        for page, pair in enumerate(doc_groups_tasks, 2):
            if (pair.classification_page1, pair.classification_id_page1) != (previous_name, previous_id):
                raise ValueError(f"Conflicting classifications for page {page - 1}")
            if pair.belongs_to_same_document:
                if (pair.classification_page1, pair.classification_id_page1) != (pair.classification_page2, pair.classification_id_page2):
                    raise ValueError("Pages in the same document must have the same classification")
                current.pages.append(page)
            else:
                result.doc_groups.append(current)
                current = DocGroup([page], pair.classification_page2, pair.classification_id_page2)
            previous_name, previous_id = pair.classification_page2, pair.classification_id_page2
        result.doc_groups.append(current)
        return result
