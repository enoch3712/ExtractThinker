import json
from typing import List, Dict, Any, Tuple
import instructor
import litellm
from instructor.exceptions import IncompleteOutputException
from litellm import Router
from pydantic_core import from_json
import re

litellm.set_verbose = True

class LLM:
    def __init__(self,
                 model: str,
                 api_base: str = None,
                 api_key: str = None,
                 api_version: str = None,
                 token_limit: int = None,
                 max_retries: int = 3):
        self.client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        self.model = model
        self.router = None
        self.api_base = api_base
        self.api_key = api_key
        self.api_version = api_version
        self.token_limit = token_limit
        self.max_retries = max_retries
        self.json_parts = []

    def load_router(self, router: Router) -> None:
        self.router = router

    def request(self, messages: List[Dict[str, str]], response_model: Any) -> Any:
        attempt = 0
        use_raw_litellm = False
        self.json_parts = []

        string_content = """{
                "title": "Tax Revenues in the EU",
                "pages": [
                    {
                    "title": "Tax Revenues Overview",
                    "number": 1,
                    "content": """ + """Figure 3: Tax revenues in the EU since 2009 (nominal terms and percentage of GDP)

7,000 41.0
6,000 40.5
5,000 40.0

a

5 4,000 3058

w (0)

© 2

2 le)

= 3,000 39.0 2
2,000 38.5
1,000 38.0

0 37.5

mmm Nominal value = == % GDP

Source: Eurostat [gov_10a_taxag], as of 31 January 2024. Nominal values converted in EUR for non-EA countries.

In 2022, total tax revenues grew below nominal GDP in 15 Member States. As shown in Figure 4,
nominal tax revenues (numerator) did not grow as fast as nominal GDP (denominator)
in many EU countries,
which led to a decrease in their tax revenue-to-GDP ratio
in 2022. The largest gaps were recorded in Poland (tax
revenues up by 9.5%,
7.1 pp below nominal GDP) and specially in Denmark (the only EU country where
tax
revenues decreased, by 2.2%, although nominal GDP surged by 11.0%). The
rates of change of tax revenues and
nominal GDP may differ for a variety of
reasons that require more detailed country level analysis. Generally
speaking,
tax systems may be distorted by high inflation through different channels, such
as non-adjustment of
nominal tax parameters or time-lag between tax payments/refunds
and liabilities incurred that may change the
real value of the payment to be
made (?*) (17). Recent research from the OECD suggests that revenues from CIT
and VAT are historically more sensitive to GDP changes than those from PIT and,
especially, those from SC and
excises (OECD, 2023a). It could be also the case
that nominal output from sectors with a lower tax burden is
growing faster
than others with a higher tax burden, or vice versa. Similarly, tax policy changes
are not
homogeneous across sectors and could impact more/less sectors affected
by higher/lower inflation.""" + """
                    }
                ]
            }"""

        # Clean the string content by removing only control characters
        string_content = ''.join(char for char in string_content if ord(char) >= 32 or char in '\n\r\t')

        # Format numbers with commas while preserving the rest of the content
        string_content = re.sub(r'(\d+),(\d{3})', r'\1\2', string_content)

        # Escape newlines and quotes to make it valid JSON
        string_content = string_content.replace('\\', '\\\\')  # escape backslashes first
        string_content = string_content.replace('"', '\\"')    # escape quotes
        string_content = string_content.replace('\n', '\\n')   # escape newlines

        # Debug print
        print("String content before parsing:", string_content)
        
        try:
            # First try to parse it as JSON
            string_content_1 = json.loads(string_content)
            print("Successfully parsed JSON")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"First few characters: {string_content[:100]}")

        m = response_model.model_validate(
            string_content_1
        )
            
        while attempt < self.max_retries:
            try:
                if use_raw_litellm:
                    # Use raw litellm for large responses
                    if self.router:
                        raw_response = self.router.completion(
                            model=self.model,
                            messages=messages
                        )
                    else:
                        raw_response = litellm.completion(
                            model=self.model,
                            messages=messages,
                            api_base=self.api_base,
                            api_key=self.api_key,
                            api_version=self.api_version,
                            max_tokens=self.token_limit or 500
                        )
                    # Cast raw response to pydantic model
                    content = raw_response.choices[0].message.content
                    self.json_parts.append(content)  # Add the content to json_parts
                    return self._process_json_response(self.json_parts, response_model)
                else:
                    # Try with instructor first
                    if self.router:
                        return self.router.completion(
                            model=self.model,
                            messages=messages,
                            response_model=response_model
                        )
                    else:
                        return self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            response_model=response_model,
                            api_base=self.api_base,
                            api_key=self.api_key,
                            api_version=self.api_version,
                            max_tokens=self.token_limit or 500,
                            max_retries=1
                        )
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}")
                actual_exception = e.args[0] if e.args else e
                
                if isinstance(actual_exception, IncompleteOutputException):
                    # Save the partial JSON response
                    partial_content = actual_exception.last_completion.choices[0].message.content
                    self.json_parts.append(partial_content)
                    
                    messages = self._adjust_prompt(
                        messages,
                        partial_content
                    )
                    use_raw_litellm = True
                    if not messages:
                        print("Cannot process the response.")
                        break
                else:
                    print(f"An error occurred: {actual_exception}")
                    break
                
            attempt += 1
        raise Exception("Failed to get a complete response after retries.")
    
    def _adjust_prompt(self, messages: List[Dict[str, str]], content: str) -> List[Dict[str, str]]:
        """
        Rebuilds messages with type-structured content for continuation requests.
        
        Examples for how to handle partial JSON responses are included in the system message below.
        These examples guide the model on how to properly merge partial JSON segments.

        The logic here takes the previously obtained partial JSON and requests the model
        to continue the JSON. The examples serve as a reference for the model.
        """
        system_content = ""
        user_content = ""
        images = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n"
            elif msg["role"] == "user":
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            user_content += item["text"] + "\n"
                        elif item.get("type") == "image_url":
                            images.append(item)
                elif isinstance(msg["content"], str):
                    user_content += msg["content"] + "\n"
        
        # Injecting the examples into the system message:
        # The structure remains the same; we just add a large block of text with the examples after the rule.
        examples_text = (
            "\n\n## Examples of Partial JSON Merging\n\n"
            "Below are examples that illustrate how to handle partial JSON responses.\n"
            "These examples show how to continue values, strings, arrays, and objects.\n"
            
            "\n### Example 1: Continuation of an Object with Nested Arrays\n"
            "**Initial partial response:**\n"
            "```json\n"
            "{\n"
            "  \"title\": \"Extra Content\",\n"
            "  \"pages\": [\n"
            "    {\n"
            "      \"title\": \"Tax Revenues\",\n"
            "      \"number\": 1,\n"
            "      \"additional_data\": [\n"
            "        {\n"
            "          \"year\": 2020,\n"
            "          \"value\":\n"
            "```\n"
            "**Continuation part:**\n"
            "```json\n"
            "3000.45\n"
            "        },\n"
            "        {\n"
            "          \"year\": 2021,\n"
            "          \"value\": 3200.10\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            
            "**Final combined JSON:**\n"
            "```json\n"
            "{\n"
            "  \"title\": \"Extra Content\",\n"
            "  \"pages\": [\n"
            "    {\n"
            "      \"title\": \"Tax Revenues\",\n"
            "      \"number\": 1,\n"
            "      \"additional_data\": [\n"
            "        {\n"
            "          \"year\": 2020,\n"
            "          \"value\": 3000.45\n"
            "        },\n"
            "        {\n"
            "          \"year\": 2021,\n"
            "          \"value\": 3200.10\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            
            "\n### Example 2: Continuation of a String Value\n"
            "**Initial partial response:**\n"
            "```json\n"
            "{\n"
            "  \"document\": {\n"
            "    \"title\": \"Report on Fiscal Measures\",\n"
            "    \"content\": \"In this report, we analyze ... tax reve\n"
            "```\n"
            "**Continuation part:**\n"
            "```json\n"
            "nues and expenditures.\"\n"
            "  }\n"
            "}\n"
            "```\n"
            
            "**Final combined JSON:**\n"
            "```json\n"
            "{\n"
            "  \"document\": {\n"
            "    \"title\": \"Report on Fiscal Measures\",\n"
            "    \"content\": \"In this report, we analyze ... tax revenues and expenditures.\"\n"
            "  }\n"
            "}\n"
            "```\n"
            
            "\n### Example 3: Continuation of an Array Value\n"
            "**Initial partial response:**\n"
            "```json\n"
            "{\n"
            "  \"countries\": [\n"
            "    {\n"
            "      \"name\": \"Denmark\",\n"
            "      \"tax_revenue\": \"2.2%\"\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"Poland\",\n"
            "      \"tax_revenue\":\n"
            "```\n"
            "**Continuation part:**\n"
            "```json\n"
            "\"9.5%\"\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"Germany\",\n"
            "      \"tax_revenue\": \"5.0%\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            
            "**Final combined JSON:**\n"
            "```json\n"
            "{\n"
            "  \"countries\": [\n"
            "    {\n"
            "      \"name\": \"Denmark\",\n"
            "      \"tax_revenue\": \"2.2%\"\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"Poland\",\n"
            "      \"tax_revenue\": \"9.5%\"\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"Germany\",\n"
            "      \"tax_revenue\": \"5.0%\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
        )

        # Add the examples right after the RULE section in the system message
        structured_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a server API that receives document information and returns specific fields in JSON format.\n\n"
                            "##Extra Content\n\n"
                            "RULE: Give me all the pages content"
                            + examples_text  # Injecting the examples here
                        )
                    }
                ]
            }
        ]
        
        user_message_content = []
        
        if user_content:
            user_message_content.append({
                "type": "text",
                "text": user_content.strip()
            })
        
        user_message_content.extend(images)
        
        # Add JSON marker
        user_message_content.append({
            "type": "text",
            "text": "\n##JSON"
        })
        
        structured_messages.append({
            "role": "user",
            "content": user_message_content
        })
        
        structured_messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })
        
        structured_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "## CONTINUE JSON"
                }
            ]
        })
        
        return structured_messages
    
    def _process_json_response(self, json_parts: List[str], response_model: Any) -> Any:
        # This remains unchanged
        processed_parts = []
        
        for content in json_parts:
            cleaned_content = content.replace('```json', '').replace('```', '').strip()
            if cleaned_content:
                processed_parts.append(cleaned_content)
        
        if not processed_parts:
            raise ValueError("No valid JSON content found in the response")
        
        combined_json = processed_parts[0]
        
        if len(processed_parts) > 1:
            for part in processed_parts[1:]:
                if combined_json.rstrip().endswith(','):
                    while part.startswith('{') or part.startswith('['):
                        part = part[1:]
                    combined_json = combined_json + part
                else:
                    if combined_json.rstrip().endswith('"'):
                        combined_json = combined_json.rstrip()[:-1] + part
                    else:
                        combined_json += part
        
        while '}}]}' in combined_json:
            combined_json = combined_json.replace('}}]}', '}]}')
        
        try:
            # First attempt: Handle common escape sequences
            try:
                # Convert \x escapes to \u escapes
                decoded_json = combined_json.encode('raw_unicode_escape').decode('utf-8')
                # Handle any remaining escape sequences
                decoded_json = decoded_json.encode('utf-8').decode('unicode_escape')
                return response_model.model_validate(decoded_json)
            except Exception as e:
                # Fallback: Try direct ASCII with replacement
                decoded_json = combined_json.encode('ascii', errors='replace').decode('ascii')
                return response_model.model_validate(decoded_json)
        except Exception as e:
            raise ValueError(f"Failed to parse combined JSON: {e}\nJSON content: {combined_json}")
