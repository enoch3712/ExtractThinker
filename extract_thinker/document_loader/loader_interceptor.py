from abc import ABC, abstractmethod


class LoaderInterceptor(ABC):
    @abstractmethod
    def process(self, file: str, content: str) -> None:
        raise NotImplementedError
