import argparse
import json
import os
from typing import Type, Dict, Any, Optional

from extract_thinker import Extractor, Contract
from extract_thinker.document_loader.document_loader import DocumentLoader
from extract_thinker.llm import LLM
from extract_thinker.eval.evaluator import Evaluator
from extract_thinker.eval.dataset import FileSystemDataset
from extract_thinker.eval.report import EvaluationReport


def load_contract(contract_path: str) -> Type[Contract]:
    """
    Dynamically load a Contract class from a Python file.
    
    Args:
        contract_path: Path to Python file containing the Contract class
        
    Returns:
        Type[Contract]: The Contract class
    """
    import importlib.util
    
    # Get the filename without extension for the module name
    module_name = os.path.basename(contract_path).split('.')[0]
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the Contract class
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, Contract) and attr != Contract:
            return attr
    
    raise ValueError(f"No Contract class found in {contract_path}")


def setup_extractor(config: Dict[str, Any]) -> Extractor:
    """
    Set up an Extractor based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Extractor: Configured extractor
    """
    # Create extractor
    extractor = Extractor()
    
    # Load document loader if specified
    if "document_loader" in config:
        loader_config = config["document_loader"]
        loader_type = loader_config["type"]
        
        # Import the loader class dynamically
        import importlib
        loader_module = importlib.import_module(f"extract_thinker.document_loader")
        loader_class = getattr(loader_module, loader_type)
        
        # Initialize loader with parameters
        loader_params = loader_config.get("params", {})
        loader = loader_class(**loader_params)
        
        # Load document loader
        extractor.load_document_loader(loader)
    
    # Load LLM
    if "llm" in config:
        llm_config = config["llm"]
        if isinstance(llm_config, str):
            # Simple case: just a model name
            extractor.load_llm(llm_config)
        else:
            # Complex case: LLM configuration
            model_name = llm_config["model"]
            params = llm_config.get("params", {})
            
            if "api_base" in params:
                # Create custom LLM
                llm = LLM(model_name, **params)
                extractor.load_llm(llm)
            else:
                # Standard LLM
                extractor.load_llm(model_name)
    
    return extractor


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Evaluate ExtractThinker extraction performance")
    parser.add_argument("--config", required=True, help="Path to evaluation configuration file")
    parser.add_argument("--output", default="eval_results.json", help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set up the extractor
    extractor = setup_extractor(config)
    
    # Load the contract class
    contract_class = load_contract(config["contract_path"])
    
    # Set up the dataset
    dataset = FileSystemDataset(
        documents_dir=config["documents_dir"],
        labels_path=config["labels_path"],
        name=config.get("dataset_name", "Custom Dataset"),
        file_pattern=config.get("file_pattern", "*.*")
    )
    
    # Create and run the evaluator
    evaluator = Evaluator(
        extractor=extractor,
        response_model=contract_class,
        vision=config.get("vision", False),
        content=config.get("content")
    )
    
    # Run evaluation
    report = evaluator.evaluate(
        dataset=dataset,
        evaluation_name=config.get("evaluation_name", "Extraction Evaluation"),
        skip_failures=config.get("skip_failures", False)
    )
    
    # Print summary
    report.print_summary()
    
    # Save report
    evaluator.save_report(report, args.output)


if __name__ == "__main__":
    main() 