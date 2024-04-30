import asyncio
from typing import List, Optional
from extract_thinker.extractor import Extractor
from extract_thinker.models import Classification
from extract_thinker.models import (
    DocGroups2,
    DocGroup,
    DocGroups,
)
import os


class Process:
    def __init__(self):
        self.extractors: List[Extractor] = []
        doc_groups: Optional[DocGroups] = None

    def add_extractor(self, extractor: Extractor):
        self.extractors.append(extractor)

    def loadSplitter(self, splitter):
        self.splitter = splitter
        return self

    def split(self, classifications: List[Classification]):
        splitter = self.splitter

        # Check if the file is a PDF
        _, ext = os.path.splitext(self.file)
        if ext.lower() != ".pdf":
            raise ValueError("Invalid file type. Only PDFs are accepted.")

        images = self.document_loader.convert_pdf_to_images(self.file)

        groups = splitter.split_document_into_groups([self.file])

        loop = asyncio.get_event_loop()
        processedGroups = loop.run_until_complete(
            splitter.process_split_groups(groups, classifications)
        )

        doc_groups = self.aggregate_split_documents_2(processedGroups)

        self.doc_groups = doc_groups

        return self

    def aggregate_split_documents_2(doc_groups_tasks: List[DocGroups2]) -> DocGroups:
        doc_groups = DocGroups()
        current_group = DocGroup()
        page_number = 1

        # do the first group outside of the loop
        doc_group = doc_groups_tasks[0]

        if doc_group.belongs_to_same_document:
            current_group.pages = [1, 2]
            current_group.classification = doc_group.classification_page1
            current_group.certainties = [
                doc_group.certainty,
                doc_groups_tasks[1].certainty,
            ]
        else:
            current_group.pages = [1]
            current_group.classification = doc_group.classification_page1
            current_group.certainties = [doc_group.certainty]

            doc_groups.doc_groups.append(current_group)

            current_group = DocGroup()
            current_group.pages = [2]
            current_group.classification = doc_group.classification_page2
            current_group.certainties = [doc_groups_tasks[1].certainty]

        page_number += 1

        for index in range(1, len(doc_groups_tasks)):
            doc_group_2 = doc_groups_tasks[index]

            if doc_group_2.belongs_to_same_document:
                current_group.pages.append(page_number + 1)
                current_group.certainties.append(doc_group_2.certainty)
            else:
                doc_groups.doc_groups.append(current_group)

                current_group = DocGroup()
                current_group.classification = doc_group_2.classification_page2
                current_group.pages = [page_number + 1]
                current_group.certainties = [doc_group_2.certainty]

            page_number += 1

        doc_groups.doc_groups.append(current_group)  # the last group

        return doc_groups

    def where(self, condition):
        pass
