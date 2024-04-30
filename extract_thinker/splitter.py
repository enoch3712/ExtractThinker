
import asyncio
from abc import ABC, abstractmethod
from typing import IO, List, Union
from extract_thinker.models import Classification, DocGroups2


class Splitter(ABC):
    @abstractmethod
    def belongs_to_same_document(self,
                                 page1: Union[str, IO],
                                 page2: Union[str, IO],
                                 contract: str) -> DocGroups2:
        pass

    def split_document_into_groups(
        self, document: List[Union[str, IO]]
    ) -> List[List[Union[str, IO]]]:
        # Assuming document is a list of pages
        page_per_split = 2
        split = []
        for i in range(0, len(document), page_per_split):
            group = document[i: i + page_per_split]
            # If last group has only one page, remove it
            if len(group) != 1:
                split.append(group)
        return split

    async def process_split_groups(self,
                                   split: List[List[Union[str, IO]]],
                                   classifications: List[Classification]
                                   ) -> List[DocGroups2]:
        tasks = [self.process_group(x, classifications) for x in split]
        doc_groups = await asyncio.gather(*tasks)
        return doc_groups

    async def process_group(self,
                            group: List[Union[str, IO]],
                            contract: str) -> DocGroups2:
        split_result = await self.belongs_to_same_document(group[0],
                                                           group[1],
                                                           contract)
        return split_result
