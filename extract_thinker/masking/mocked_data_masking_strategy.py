import re
from extract_thinker.llm import LLM
from extract_thinker.masking.abstract_masking_strategy import AbstractMaskingStrategy
from extract_thinker.models.MaskContract import MaskContract

class MockedDataMaskingStrategy(AbstractMaskingStrategy):
    def __init__(self, llm: LLM):
        super().__init__(llm)

    async def mask_content(self, content: str) -> MaskContract:
        # Step 1: Get masked text and mocked data mapping
        messages_step1 = [
            {
                "role": "system",
                "content": "You are an AI assistant that masks sensitive information in text with mocked data."
            },
            {
                "role": "user",
                "content": f"""
                Please mask all sensitive information in the following text with mocked data. Replace sensitive information with realistic but fake data. Return the masked text and a mapping of original values to mocked data.
                - Keep all values, doesnt constitute sensitive information

                Here are some examples:

                Example 1:
                Original text:
                John Smith lives at 123 Main St, New York, NY 10001. His phone number is (555) 123-4567.

                Mocked data mapping:
                "John Smith": "Michael Johnson"
                "123 Main St, New York, NY 10001": "456 Oak Ave, Chicago, IL 60601"
                "(555) 123-4567": "(312) 555-7890"

                Masked text:
                Michael Johnson lives at 456 Oak Ave, Chicago, IL 60601. His phone number is (312) 555-7890.

                Example 2:
                Original text:
                Sarah Johnson ordered a laptop from TechStore on 2023-05-15. Her email is sarah.j@email.com.

                Mocked data mapping:
                "Sarah Johnson": "Emma Thompson"
                "laptop": "tablet"
                "TechStore": "GadgetWorld"
                "2023-05-15": "2023-06-22"
                "sarah.j@email.com": "emma.t@fakemail.com"

                Masked text:
                Emma Thompson ordered a tablet from GadgetWorld on 2023-06-22. Her email is emma.t@fakemail.com.

                Now, please mask the following text:

                Text to mask:
                {content}

                Give me the mocked data mapping with the original value and respective mocked data, and then the Masked text with the mocked data.
                """
            }
        ]

        response_step1 = self.llm.request(messages_step1)

        response_step1_content = response_step1.choices[0].message.content

        # Step 2: Convert to JSON format
        messages_step2 = [
            {
                "role": "system",
                "content": "You are an AI assistant that converts masked text information into JSON format."
            },
            {
                "role": "user",
                "content": f"""
                Convert the following masked texts and mocked data mappings into a JSON format. For each example, the JSON should have two main keys: "mapping" (a dictionary of original values and their mocked data) and "masked_text" (the text with mocked data).

                Example 1:
                Mocked data mapping:
                "John Smith": "Michael Johnson"
                "123 Main St, New York, NY 10001": "456 Oak Ave, Chicago, IL 60601"
                "(555) 123-4567": "(312) 555-7890"

                Masked text:
                Michael Johnson lives at 456 Oak Ave, Chicago, IL 60601. His phone number is (312) 555-7890.

                Output:
                {{
                    "mapping": {{
                        "John Smith": "Michael Johnson",
                        "123 Main St, New York, NY 10001": "456 Oak Ave, Chicago, IL 60601",
                        "(555) 123-4567": "(312) 555-7890"
                    }},
                    "masked_text": "Michael Johnson lives at 456 Oak Ave, Chicago, IL 60601. His phone number is (312) 555-7890."
                }}

                Example 2:
                Mocked data mapping:
                "Sarah Johnson": "Emma Thompson"
                "laptop": "tablet"
                "TechStore": "GadgetWorld"
                "2023-05-15": "2023-06-22"
                "sarah.j@email.com": "emma.t@fakemail.com"

                Masked text:
                Emma Thompson ordered a tablet from GadgetWorld on 2023-06-22. Her email is emma.t@fakemail.com.

                Output:
                {{
                    "mapping": {{
                        "Sarah Johnson": "Emma Thompson",
                        "laptop": "tablet",
                        "TechStore": "GadgetWorld",
                        "2023-05-15": "2023-06-22",
                        "sarah.j@email.com": "emma.t@fakemail.com"
                    }},
                    "masked_text": "Emma Thompson ordered a tablet from GadgetWorld on 2023-06-22. Her email is emma.t@fakemail.com."
                }}

                Now, please convert the following masked text and mocked data mapping into JSON format:

                {response_step1_content}

                ##JSON
                """
            }
        ]

        response_step2 = self.llm.request(messages_step2, MaskContract)

        masked_text = response_step2.masked_text
        mapping = response_step2.mapping

        for original, mocked in mapping.items():
            if original in masked_text:
                masked_text = masked_text.replace(original, mocked)

        response_step2.masked_text = masked_text

        return response_step2

    def unmask_content(self, masked_content: str, mapping: dict) -> str:
        for mocked, original in mapping.items():
            masked_content = masked_content.replace(mocked, original)
        return masked_content