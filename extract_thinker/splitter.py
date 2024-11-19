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

    def aggregate_doc_groups(self, doc_groups_tasks: List[DocGroups2]) -> DocGroups:
        """
        Aggregate the results from belongs_to_same_document comparisons into final document groups.
        This is the base implementation that can be used by all splitter implementations.
        """
        doc_groups = DocGroups()
        current_group = DocGroup(pages=[], classification="")
        page_number = 1

        if not doc_groups_tasks:
            return doc_groups

        # Handle the first group
        doc_group = doc_groups_tasks[0]
        if doc_group.belongs_to_same_document:
            current_group.pages = [1, 2]
            current_group.classification = doc_group.classification_page1
        else:
            # First page is its own document
            current_group.pages = [1]
            current_group.classification = doc_group.classification_page1
            doc_groups.doc_groups.append(current_group)
            
            # Start new group with second page
            current_group = DocGroup(pages=[2], classification=doc_group.classification_page2)

        page_number += 1

        # Process remaining groups
        for doc_group in doc_groups_tasks[1:]:
            if doc_group.belongs_to_same_document:
                current_group.pages.append(page_number + 1)
            else:
                doc_groups.doc_groups.append(current_group)
                current_group = DocGroup(
                    pages=[page_number + 1],
                    classification=doc_group.classification_page2
                )
            page_number += 1

        # Add the last group
        doc_groups.doc_groups.append(current_group)

        return doc_groups