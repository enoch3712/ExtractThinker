import asyncio
from typing import IO, Any, Dict, List, Optional, Union
from extract_thinker.extractor import Extractor
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

class ClassificationStrategy(Enum):
    CONSENSUS = "consensus"
    HIGHER_ORDER = "higher_order"
    CONSENSUS_WITH_THRESHOLD = "both"


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
        self.masking_llm: Optional[LLM] = None
        self.masking_enabled: bool = False

    def add_masking_llm(self, model: Optional[str] = None) -> None:
        self.masking_enabled = True

        if isinstance(model, LLM):
            self.masking_llm = model
        elif model is not None:
            self.masking_llm = LLM(model)
        else:
            raise ValueError("Either a model string or an LLM object must be provided.")
        
    async def mask_content(self, content: str) -> MaskContract:
        if not self.masking_enabled or not self.masking_llm:
            raise ValueError("Masking is not enabled, please set a masking llm with add_masking_llm")

        # Step 1: Get masked text and placeholder list
        messages_step1 = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that masks sensitive information in text."
                },
                {
                    "role": "user",
                    "content": f"""
                    Please mask all sensitive information in the following text. Replace sensitive information with placeholders like [PERSON1], [PERSON2], [ADDRESS1], [ADDRESS2], [PHONE1], [PHONE2], etc. Return the masked text and a list of placeholders with their original values.

                    Here are some examples:

                    Example 1:
                    Original text:
                    John Smith lives at 123 Main St, New York, NY 10001. His phone number is (555) 123-4567.

                    Placeholder list:
                    [PERSON1]: John Smith
                    [ADDRESS1]: 123 Main St, New York, NY 10001
                    [PHONE1]: (555) 123-4567

                    Masked text:
                    [PERSON1] lives at [ADDRESS1]. His phone number is [PHONE1].

                    Example 2:
                    Original text:
                    Sarah Johnson ordered a laptop from TechStore on 2023-05-15. Her email is sarah.j@email.com.

                    Placeholder list:
                    [PERSON1]: Sarah Johnson
                    [PRODUCT1]: laptop
                    [STORE1]: TechStore
                    [DATE1]: 2023-05-15
                    [EMAIL1]: sarah.j@email.com

                    Masked text:
                    [PERSON1] ordered a [PRODUCT1] from [STORE1] on [DATE1]. Her email is [EMAIL1].

                    Example 3:
                    Original text:
                    Dr. Emily Brown, born on 1985-03-22, works at Central Hospital. Her patient, Mr. David Lee, has an appointment on 2023-06-10 at 2:30 PM.

                    Placeholder list:
                    [PERSON1]: Dr. Emily Brown
                    [DATE1]: 1985-03-22
                    [HOSPITAL1]: Central Hospital
                    [PERSON2]: Mr. David Lee
                    [DATE2]: 2023-06-10
                    [TIME1]: 2:30 PM

                    Masked text:
                    [PERSON1], born on [DATE1], works at [HOSPITAL1]. Her patient, [PERSON2], has an appointment on [DATE2] at [TIME1].

                    Now, please mask the following text:

                    Text to mask:
                    {content}

                    Give me the placeholder list with the value and respective placeholder, and then the Masked text with the placeholders.
                    """
                }
            ]

        response_step1 = self.masking_llm.request(messages_step1)

        response_step1_content = response_step1.choices[0].message.content

        # Step 2: Convert to JSON format
        messages_step2 = [
            {
                "role": "system",
                "content": "You are an AI assistant that converts masked text information into JSON format."
            },
            {
                "role": "user",
                "content": f"""
                Convert the following masked texts and placeholder lists into a JSON format. For each example, the JSON should have two main keys: "mapping" (a dictionary of placeholders and their original values) and "masked_text" (the text with placeholders).
                Always use [], not "" or ''
                Make sure that masked_text contains no sensitive information, only the placeholders.

                Example 1:
                Placeholder list:
                [PERSON1]: John Smith
                [ADDRESS1]: 123 Main St, New York, NY 10001
                [PHONE1]: (555) 123-4567

                Masked text:
                [PERSON1] lives at [ADDRESS1]. His phone number is [PHONE1].

                Output:
                {{
                    "mapping": {{
                        "[PERSON1]": "John Smith",
                        "[ADDRESS1]": "123 Main St, New York, NY 10001",
                        "[PHONE1]": "(555) 123-4567"
                    }},
                    "masked_text": "[PERSON1] lives at [ADDRESS1]. His phone number is [PHONE1]."
                }}

                Example 2:
                Placeholder list:
                [PERSON1]: Sarah Johnson
                [PRODUCT1]: laptop
                [STORE1]: TechStore
                [DATE1]: 2023-05-15
                [EMAIL1]: sarah.j@email.com

                Masked text:
                [PERSON1] ordered a [PRODUCT1] from [STORE1] on [DATE1]. Her email is [EMAIL1].

                Output:
                {{
                    "mapping": {{
                        "[PERSON1]": "Sarah Johnson",
                        "[PRODUCT1]": "laptop",
                        "[STORE1]": "TechStore",
                        "[DATE1]": "2023-05-15",
                        "[EMAIL1]": "sarah.j@email.com"
                    }},
                    "masked_text": "[PERSON1] ordered a [PRODUCT1] from [STORE1] on [DATE1]. Her email is [EMAIL1]."
                }}

                Example 3:
                Placeholder list:
                [PERSON1]: Dr. Emily Brown
                [DATE1]: 1985-03-22
                [HOSPITAL1]: Central Hospital
                [PERSON2]: Mr. David Lee
                [DATE2]: 2023-06-10
                [TIME1]: 2:30 PM

                Masked text:
                [PERSON1], born on [DATE1], works at [HOSPITAL1]. Her patient, [PERSON2], has an appointment on [DATE2] at [TIME1].

                Output:
                {{
                    "mapping": {{
                        "[PERSON1]": "Dr. Emily Brown",
                        "[DATE1]": "1985-03-22",
                        "[HOSPITAL1]": "Central Hospital",
                        "[PERSON2]": "Mr. David Lee",
                        "[DATE2]": "2023-06-10",
                        "[TIME1]": "2:30 PM"
                    }},
                    "masked_text": "[PERSON1], born on [DATE1], works at [HOSPITAL1]. Her patient, [PERSON2], has an appointment on [DATE2] at [TIME1]."
                }}

                Example 4:
                Placeholder list:
                [COMPANY1]: Company XYZ
                [PERSON1]: Jane Doe
                [COMPANY2]: ABC Corp
                [DATE1]: July 1, 2023
                [AMOUNT1]: $500 million

                Masked text:
                [COMPANY1]'s CEO, [PERSON1], announced a merger with [COMPANY2] on [DATE1]. The deal is valued at [AMOUNT1].

                Output:
                {{
                    "mapping": {{
                        "[COMPANY1]": "Company XYZ",
                        "[PERSON1]": "Jane Doe",
                        "[COMPANY2]": "ABC Corp",
                        "[DATE1]": "July 1, 2023",
                        "[AMOUNT1]": "$500 million"
                    }},
                    "masked_text": "[COMPANY1]'s CEO, [PERSON1], announced a merger with [COMPANY2] on [DATE1]. The deal is valued at [AMOUNT1]."
                }}

                Example 5:
                Placeholder list:
                [CREDITCARD1]: 4111-1111-1111-1111
                [PERSON1]: Michael Johnson
                [DATE1]: 12/25
                [CVV1]: 123

                Masked text:
                The credit card number [CREDITCARD1] belongs to [PERSON1], expiring on [DATE1], with CVV [CVV1].

                Output:
                {{
                    "mapping": {{
                        "[CREDITCARD1]": "4111-1111-1111-1111",
                        "[PERSON1]": "Michael Johnson",
                        "[DATE1]": "12/25",
                        "[CVV1]": "123"
                    }},
                    "masked_text": "The credit card number [CREDITCARD1] belongs to [PERSON1], expiring on [DATE1], with CVV [CVV1]."
                }}

                Now, please convert the following masked text and placeholder list into JSON format:

                {response_step1_content}

                ##JSON
                """
            }
        ]

        response_step2 = self.masking_llm.request(messages_step2, MaskContract)
        
        return response_step2
    
    def unmask_content(self, masked_content: str, mapping: dict) -> str:
        for placeholder, original in mapping.items():
            masked_content = masked_content.replace(placeholder, original)
        return masked_content

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
