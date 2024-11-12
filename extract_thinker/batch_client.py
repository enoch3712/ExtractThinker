from extract_thinker.models.batch_result import BatchResult
from extract_thinker.models.batch_status import BatchStatus
from abc import ABC, abstractmethod
from typing import List

class BatchClient(ABC):
    @abstractmethod
    def create_batch(self, batch_requests: List[dict]) -> str:
        """Create a batch processing job and return the batch ID."""
        pass

    @abstractmethod
    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """Retrieve the status of the batch job."""
        pass

    @abstractmethod
    def get_batch_results(self, batch_id: str) -> BatchResult:
        """Retrieve the results of the completed batch job."""
        pass
