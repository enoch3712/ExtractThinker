import asyncio
from typing import IO, Any, Dict, List, Optional, Union
from extract_thinker.extractor import Extractor
from extract_thinker.masking.deterministic_hashing_masking_strategy import DeterministicHashingMaskingStrategy
from extract_thinker.models.classification import Classification
from extract_thinker.document_loader.document_loader import DocumentLoader
from extract_thinker.models.classification_tree import ClassificationTree
from extract_thinker.models.classification_node import ClassificationNode
from extract_thinker.models.doc_group import DocGroup
from extract_thinker.models.doc_groups2 import DocGroups2
from extract_thinker.splitter import Splitter
from extract_thinker.models.doc_groups import (
    DocGroups,
)
from extract_thinker.utils import get_image_type
from extract_thinker.llm import LLM
from extract_thinker.models.MaskContract import MaskContract
from enum import Enum
from extract_thinker.masking.llm_masking_strategy import LLMMaskingStrategy
from extract_thinker.masking.simple_placeholder_masking_strategy import SimplePlaceholderMaskingStrategy
from extract_thinker.masking.mocked_data_masking_strategy import MockedDataMaskingStrategy
from extract_thinker.masking.abstract_masking_strategy import AbstractMaskingStrategy

class ClassificationStrategy(Enum):
    CONSENSUS = "consensus"
    HIGHER_ORDER = "higher_order"
    CONSENSUS_WITH_THRESHOLD = "both"

class MaskingStrategy(Enum):
    SIMPLE_PLACEHOLDER = "simple_placeholder"
    MOCKED_DATA = "mocked_data"
    DETERMINISTIC_HASHING = "deterministic_hashing"

class Process:
    def __init__(self):
        # self.extractors: List[Extractor] = []
        self.doc_groups: Optional[DocGroups] = None
        self.split_classifications: List[Classification] = []
        self.extractor_groups: List[List[Extractor]] = []  # for classication
        self.document_loaders_by_file_type: Dict[str, DocumentLoader] = {}
        self.document_loader: Optional[DocumentLoader] = None
        self.file_path: Optional[str] = None
        self.file_stream: Optional[IO] = None
        self.splitter: Optional[Splitter] = None
        self.masking_strategy: Optional[AbstractMaskingStrategy] = None
        self.llm: Optional[LLM] = None

    def add_masking_llm(self, model: Optional[str] = None, strategy: Optional[MaskContract] = MaskingStrategy.SIMPLE_PLACEHOLDER):
        if isinstance(model, LLM):
            self.llm = model
        elif model is not None:
            self.llm = LLM(model)
            
        if strategy == MaskingStrategy.SIMPLE_PLACEHOLDER:
            self.masking_strategy = SimplePlaceholderMaskingStrategy(self.llm)
        elif strategy == MaskingStrategy.MOCKED_DATA:
            self.masking_strategy = MockedDataMaskingStrategy(self.llm)
        elif strategy == MaskingStrategy.DETERMINISTIC_HASHING:
            self.masking_strategy = DeterministicHashingMaskingStrategy(self.llm)

    async def mask_content(self, content: str) -> MaskContract:
        if self.masking_strategy is None:
            raise ValueError("No masking strategy has been set. Please set a masking strategy with add_masking_strategy.")

        return await self.masking_strategy.mask_content(content)

    def unmask_content(self, masked_content: str, mapping: dict) -> str:
        if self.masking_strategy is None:
            raise ValueError("No masking strategy has been set. Please set a masking strategy with add_masking_strategy.")

        return self.masking_strategy.unmask_content(masked_content, mapping)

    def set_document_loader_for_file_type(self, file_type: str, document_loader: DocumentLoader):
        if self.document_loader is not None:
            raise ValueError("Cannot set a document loader for a specific file type when a default loader is already set.")
        self.document_loaders_by_file_type[file_type] = document_loader

    def load_document_loader(self, document_loader: DocumentLoader):
        if self.document_loaders_by_file_type:
            raise ValueError("Cannot set a default document loader when specific loaders are already set.")
        self.document_loader = document_loader
        return self

    def load_splitter(self, splitter: Splitter):
        self.splitter = splitter
        return self

    def add_classify_extractor(self, extractor_groups: List[List[Extractor]]):
        for extractors in extractor_groups:
            self.extractor_groups.append(extractors)
        return self

    async def _classify_async(self, extractor: Extractor, file: str, classifications: List[Classification], image: bool = False):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extractor.classify, file, classifications, image)

    def classify(self, file: str, classifications, strategy: ClassificationStrategy = ClassificationStrategy.CONSENSUS, threshold: int = 9, image: bool = False) -> Optional[Classification]:
        result = asyncio.run(self.classify_async(file, classifications, strategy, threshold, image))

        return result

    async def classify_async(
        self,
        file: str,
        classifications: Union[List[Classification], ClassificationTree],
        strategy: ClassificationStrategy = ClassificationStrategy.CONSENSUS,
        threshold: int = 9,
        image: str = False
    ) -> Optional[Classification]:

        if isinstance(classifications, ClassificationTree):
            return await self._classify_tree_async(file, classifications, threshold, image)

        for extractor_group in self.extractor_groups:
            group_classifications = await asyncio.gather(*(self._classify_async(extractor, file, classifications, image) for extractor in extractor_group))

        # Implement different strategies
        if strategy == ClassificationStrategy.CONSENSUS:
            # Check if all classifications in the group are the same
            if len(set(group_classifications)) == 1:
                return group_classifications[0]
        elif strategy == ClassificationStrategy.HIGHER_ORDER:
            # Pick the result with the highest confidence
            return max(group_classifications, key=lambda c: c.confidence)
        elif strategy == ClassificationStrategy.CONSENSUS_WITH_THRESHOLD:
            if len(set(group_classifications)) == 1:
                maxResult = max(group_classifications, key=lambda c: c.confidence)
                if maxResult.confidence >= threshold:
                    return maxResult

        raise ValueError("No consensus could be reached on the classification of the document. Please try again with a different strategy or threshold.")

    async def _classify_tree_async(
        self, 
        file: str, 
        classification_tree: ClassificationTree, 
        threshold: float,
        image: bool
    ) -> Optional[Classification]:
        """
        Perform classification in a hierarchical, level-by-level approach.
        """
        best_classification = None
        current_nodes = classification_tree.nodes

        while current_nodes:
            # Get the list of classifications at the current level
            classifications = [node.classification for node in current_nodes]

            # Classify among the current level's classifications
            classification = await self._classify_async(
                extractor=self.extractor_groups[0][0],
                file=file, 
                classifications=classifications, 
                image=image
            )

            if classification.confidence < threshold:
                raise ValueError(
                    f"Classification confidence {classification.confidence} "
                    f"for '{classification.classification}' is below the threshold of {threshold}."
                )

            best_classification = classification

            matching_node = next(
                (
                    node for node in current_nodes 
                    if node.classification.name == best_classification.name
                ),
                None
            )

            if matching_node is None:
                raise ValueError(
                    f"No matching node found for classification '{classification.classification}'."
                )

            if matching_node.children:
                current_nodes = matching_node.children
            else:
                break

        return best_classification

    async def classify_extractor(self, session, extractor, file):
        return await session.run(extractor.classify, file)

    # check if there is only the default one, if not, get from the file type. if none is present, raise an error
    def get_document_loader(self, file):
        if self.document_loader is not None:
            return self.document_loader

        filetype = get_image_type(file)
        return self.document_loaders_by_file_type.get(filetype, None)

    def load_file(self, file):
        self.file_path = file
        return self

    def split(self, classifications: List[Classification]):

        self.split_classifications = classifications

        documentLoader = self.get_document_loader(self.file_path)

        if documentLoader is None:
            raise ValueError("No suitable document loader found for file type")

        if self.file_path:
            content = documentLoader.load_content_from_file_list(self.file_path)
        elif self.file_stream:
            content = documentLoader.load_content_from_stream_list(self.file_stream)
        else:
            raise ValueError("No file or stream available")

        if len(content) == 1:
            raise ValueError("Document must have at least 2 pages")

        groups = self.splitter.split_document_into_groups(content)

        loop = asyncio.get_event_loop()
        processedGroups = loop.run_until_complete(
            self.splitter.process_split_groups(groups, classifications)
        )

        doc_groups = self.aggregate_split_documents_2(processedGroups)

        # doc_groups = DocGroups()
        # doc_groups.doc_groups.append(DocGroup(pages=[1], classification='Invoice'))
        # doc_groups.doc_groups.append(DocGroup(pages=[2], classification='Invoice'))
        # doc_groups.doc_groups.append(DocGroup(pages=[3], classification='Invoice'))
        # doc_groups.doc_groups.append(DocGroup(pages=[4, 5], classification='Invoice'))

        self.doc_groups = doc_groups

        return self

    def aggregate_split_documents_2(self, doc_groups_tasks: List[DocGroups2]) -> DocGroups:
        doc_groups = DocGroups()
        current_group = DocGroup()
        page_number = 1

        # do the first group outside of the loop
        doc_group = doc_groups_tasks[0]

        if doc_group.belongs_to_same_document:
            current_group.pages = [1, 2]
            current_group.classification = doc_group.classification_page1
        else:
            current_group.pages = [1]
            current_group.classification = doc_group.classification_page1

            doc_groups.doc_groups.append(current_group)

            current_group = DocGroup()
            current_group.pages = [2]
            current_group.classification = doc_group.classification_page2

        page_number += 1

        for index in range(1, len(doc_groups_tasks)):
            doc_group_2 = doc_groups_tasks[index]

            if doc_group_2.belongs_to_same_document:
                current_group.pages.append(page_number + 1)
            else:
                doc_groups.doc_groups.append(current_group)

                current_group = DocGroup()
                current_group.classification = doc_group_2.classification_page2
                current_group.pages = [page_number + 1]

            page_number += 1

        doc_groups.doc_groups.append(current_group)  # the last group

        return doc_groups

    def where(self, condition):
        pass

    def extract(self) -> List[Any]:
        if self.doc_groups is None:
            raise ValueError("Document groups have not been initialized")

        async def _extract(doc_group):
            classificationStr = doc_group.classification  # str

            for classification in self.split_classifications:
                if classification.name == classificationStr:
                    extractor = classification.extractor
                    contract = classification.contract
                    break

            if extractor is None:
                raise ValueError("Extractor not found for classification")

            documentLoader = self.get_document_loader(self.file_path)

            if documentLoader is None:
                raise ValueError("No suitable document loader found for file type")

            if self.file_path:
                content = documentLoader.load_content_from_file_list(self.file_path)
            elif self.file_stream:
                content = documentLoader.load_content_from_stream_list(self.file_stream)
            else:
                raise ValueError("No file or stream available")

            # doc_groups contains e.g [1,2], [3], [4,5] and doc_group is e.g [1,2]
            # content is a list of pages with the content of each page
            # get the content of the pages, add them together and extract the data

            pages_content = [content[i - 1] for i in doc_group.pages]
            return await extractor.extract_async(pages_content, contract)

        doc_groups = self.doc_groups.doc_groups

        async def process_doc_groups(groups: List[Any]) -> List[Any]:
            # Create asynchronous tasks for processing each group
            tasks = [_extract(group) for group in groups]
            try:
                # Execute all tasks concurrently and wait for all to complete
                processedGroups = await asyncio.gather(*tasks)
                return processedGroups
            except Exception as e:
                # Handle possible exceptions that might occur during task execution
                print(f"An error occurred: {e}")
                raise

        loop = asyncio.get_event_loop()
        processedGroups = loop.run_until_complete(
            process_doc_groups(doc_groups)
        )

        return processedGroups