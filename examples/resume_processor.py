import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field
import yaml

from extract_thinker import Extractor, Contract, DocumentLoaderPyPdf
from litellm import Router

from extract_thinker import LLM


def json_to_yaml(json_dict):
    # Check if json_dict is a dictionary
    if not isinstance(json_dict, dict):
        raise ValueError("json_dict must be a dictionary")

    # Convert the Python dictionary to YAML
    yaml_str = yaml.dump(json_dict)

    return yaml_str


class RoleContract(Contract):
    company_name: str = Field("Company name")
    years_of_experience: int = Field("Years of experience required. If not mention, calculate with start date and end date")
    is_remote: bool = Field("Is the role remote?")
    country: str = Field("Country of the role")
    city: Optional[str] = Field("City of the role")
    list_of_skills: List[str] = Field("""
                                          list of strings, e.g ["5 years experience", "3 years in React", "Typescript"]
                                          Make the lists of skills to be a yes/no list, so it can be used in the LLM model as a list of true/false
                                          """)


class ResumeContract(Contract):
    name: str = Field("First and Last Name")
    age: Optional[str] = Field("Age with format DD/MM/YYYY. Empty if not available")
    email: str = Field("Email address")
    phone: Optional[str] = Field("Phone number")
    address: Optional[str] = Field("Address")
    city: Optional[str] = Field("City")
    total_experience: int = Field("Total experience in years")
    can_go_to_office: Optional[bool] = Field("Can go to office. If city/location is not provider, is false. If is the same city, is true")
    list_of_skills: List[bool] = Field("Takes the list of skills and returns a list of true/false, if the candidate has that skill. E.g. ['Python', 'JavaScript', 'React', 'Node.js'] -> [True, True, False, True]")


class Person(Contract):
    name: str = Field("First and Last Name")
    list_of_skills: List[str]

load_dotenv()
cwd = os.getcwd()


def config_router():
    rpm = 5000  # Rate limit in requests per minute

    model_list = [
        {
            "model_name": "Meta-Llama-3-8B-Instruct",
            "litellm_params": {
                "model": "deepinfra/meta-llama/Meta-Llama-3-8B-Instruct",
                "api_key": os.getenv("DEEPINFRA_API_KEY"),
                "rpm": rpm,
            },
        },
        {
            "model_name": "Mistral-7B-Instruct-v0.2",
            "litellm_params": {
                "model": "deepinfra/mistralai/Mistral-7B-Instruct-v0.2",
                "api_key": os.getenv("DEEPINFRA_API_KEY"),
                "rpm": rpm,
            }
        },
        {
            "model_name": "groq-llama3-8b-8192",
            "litellm_params": {
                "model": "groq/llama3-8b-8192",
                "api_key": os.getenv("GROQ_API_KEY"),
                "rpm": rpm,
            }
        },
    ]

    # Adding fallback models
    fallback_models = [
        {
            "model_name": "claude-3-haiku-20240307",
            "litellm_params": {
                "model": "claude-3-haiku-20240307",
                "api_key": os.getenv("CLAUDE_API_KEY"),
            }
        },
        {
            "model_name": "azure-deployment",
            "litellm_params": {
                "model": "azure/<your-deployment-name>",
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_key": os.getenv("AZURE_API_KEY"),
                "rpm": 1440,
            }
        }
    ]

    # Combine the lists
    model_list.extend(fallback_models)

    # Define the router configuration
    router = Router(
        model_list=model_list,
        default_fallbacks=["claude-3-haiku-20240307", "azure/<your-deployment-name>"],
        context_window_fallbacks=[
            {"Meta-Llama-3-8B-Instruct": ["claude-3-haiku-20240307"]},
            {"groq-llama3-8b-8192": ["claude-3-haiku-20240307"]},
            {"Mistral-7B-Instruct-v0.2": ["claude-3-haiku-20240307"]}
        ],
        set_verbose=True
    )

    return router


job_role_path = os.path.join(cwd, "examples", "files", "Job_Offer.pdf")

extractor_job_role = Extractor()

extractor_job_role.load_document_loader(
    DocumentLoaderPyPdf()
)

extractor_job_role.load_llm("gpt-4o")
role_result = extractor_job_role.extract(job_role_path, RoleContract)

print(role_result.model_dump_json())

extractor_candidate = Extractor()
extractor_candidate.load_document_loader(
    DocumentLoaderPyPdf()
)

llm = LLM("groq/llama3-8b-8192")  # default model
#llm.load_router(config_router())  # load the router

extractor_candidate.load_llm(llm)

resume_content_path = os.path.join(cwd, "examples", "files", "CV_Candidate.pdf")

job_role_content = "This is the job content to be mapped: \n" + json_to_yaml(json.loads(role_result.model_dump_json()))

result = extractor_candidate.extract(resume_content_path,
                                     ResumeContract,
                                     content=job_role_content)

print(result.model_dump_json())
