from enum import Enum


class CompletionStrategy(Enum):
    CONCATENATE = "concatenate"
    PAGINATE = "paginate"
    FORBIDDEN = "forbidden"