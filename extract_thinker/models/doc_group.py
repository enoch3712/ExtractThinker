from typing import List


class DocGroup:
    def __init__(self, pages: List[int], classification: str):
        self.pages = pages
        self.classification = classification


class DocGroups:
    def __init__(self):
        self.doc_groups: List[DocGroup] = []