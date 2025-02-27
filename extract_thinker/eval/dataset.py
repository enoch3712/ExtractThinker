from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Tuple, List, Optional
import os
import json
import glob


class EvaluationDataset(ABC):
    """
    Abstract base class for evaluation datasets.
    
    A dataset provides document sources and their expected extraction results.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the dataset.
        
        Args:
            name: Optional name for the dataset
        """
        self.name = name or "Custom Dataset"
    
    @abstractmethod
    def items(self) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        """
        Iterate through the dataset items.
        
        Yields:
            Tuple containing:
                - document ID (str)
                - document source (path or content)
                - expected extraction result (dict)
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items in the dataset.
        
        Returns:
            int: Number of items
        """
        pass


class FileSystemDataset(EvaluationDataset):
    """
    A dataset that loads documents from the filesystem and expected results from JSON.
    """
    
    def __init__(
        self,
        documents_dir: str,
        labels_path: str,
        name: Optional[str] = None,
        file_pattern: str = "*.*"
    ):
        """
        Initialize the filesystem dataset.
        
        Args:
            documents_dir: Directory containing the documents to evaluate
            labels_path: Path to JSON file with expected outputs
            name: Optional name for the dataset
            file_pattern: Glob pattern to match document files
        """
        super().__init__(name)
        self.documents_dir = documents_dir
        self.labels_path = labels_path
        self.file_pattern = file_pattern
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
            
        # Get list of document files
        self.document_paths = glob.glob(os.path.join(documents_dir, file_pattern))
        
        # Validate that labels exist for all documents
        self._validate_documents()
    
    def _validate_documents(self):
        """
        Validate that labels exist for all documents and vice versa.
        
        Raises:
            ValueError: If documents are missing labels or labels are missing documents
        """
        # Check for documents without labels
        missing_labels = []
        for doc_path in self.document_paths:
            doc_id = os.path.basename(doc_path)
            if doc_id not in self.labels:
                missing_labels.append(doc_id)
        
        if missing_labels:
            raise ValueError(f"Missing labels for documents: {', '.join(missing_labels)}")
        
        # Check for labels without documents
        missing_docs = []
        for doc_id in self.labels:
            if not any(os.path.basename(path) == doc_id for path in self.document_paths):
                missing_docs.append(doc_id)
                
        if missing_docs:
            raise ValueError(f"Missing documents for labels: {', '.join(missing_docs)}")
    
    def items(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """
        Iterate through document paths and their expected extraction results.
        
        Yields:
            Tuple containing:
                - document ID (filename)
                - document path (str)
                - expected extraction result (dict)
        """
        for doc_path in self.document_paths:
            doc_id = os.path.basename(doc_path)
            expected = self.labels[doc_id]
            yield doc_id, doc_path, expected
    
    def __len__(self) -> int:
        """
        Get the number of documents in the dataset.
        
        Returns:
            int: Number of documents
        """
        return len(self.document_paths) 