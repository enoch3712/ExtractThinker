from abc import ABC, abstractmethod


class LlmInterceptor(ABC):
    @abstractmethod
    def process(self, messages: list, response: str) -> None:
        pass
