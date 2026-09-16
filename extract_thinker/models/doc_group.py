from typing import List


class DocGroup:
    def __init__(self, pages: List[int], classification: str, classification_id: int = None):
        self.pages = pages
        self.classification = classification
        self.classification_id = classification_id


class DocGroups:
    def __init__(self):
        self.doc_groups: List[DocGroup] = []