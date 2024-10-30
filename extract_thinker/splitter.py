import asyncio
from typing import Any, List
from abc import ABC, abstractmethod

from extract_thinker.models.classification import Classification
from extract_thinker.models.doc_group import DocGroups
from extract_thinker.models.doc_groups2 import DocGroups2
from extract_thinker.models.eager_doc_group import EagerDocGroup


class Splitter(ABC):
    @abstractmethod
    def belongs_to_same_document(self, page1: Any, page2: Any, contract: str) -> DocGroups2:
        pass

    @abstractmethod
    def split_lazy_doc_group(self, lazy_doc_group: List[Any], classifications: List[Classification]) -> EagerDocGroup:
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
