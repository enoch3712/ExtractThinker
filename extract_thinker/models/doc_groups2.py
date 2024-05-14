from dataclasses import dataclass


@dataclass
class DocGroups2:
    # certainty: float
    belongs_to_same_document: bool
    classification_page1: str
    classification_page2: str

    def __eq__(self, other):
        if not isinstance(other, DocGroups2):
            return NotImplemented
        return id(self) == id(other)

    def __hash__(self):
        return id(self)
