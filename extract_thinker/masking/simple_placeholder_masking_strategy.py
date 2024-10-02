import re
from extract_thinker.llm import LLM
from extract_thinker.masking.abstract_masking_strategy import AbstractMaskingStrategy
from extract_thinker.models.MaskContract import MaskContract

class SimplePlaceholderMaskingStrategy(AbstractMaskingStrategy):
    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.placeholder_counter = {}

    async def mask_content(self, content: str) -> MaskContract:
        # Step 1: Get masked text and placeholder list
        messages_step1 = [
            {
                "role": "system",
                "content": "You are an AI assistant that masks sensitive information in text."
            },
            {
                "role": "user",
                "content": f"""
                Please mask all sensitive information in the following text. Replace sensitive information with placeholders like [PERSON1], [PERSON2], [ADDRESS1], [ADDRESS2], [PHONE1], [PHONE2], etc. Return the masked text and a list of placeholders with their original values.
                - Keep all values, doesnt constitute sensitive information
                Here are some examples:

                Example 1:
                Original text:
                John Smith lives at 123 Main St, New York, NY 10001. His phone number is (555) 123-4567.

                Placeholder list:
                [PERSON1]: John Smith
                [ADDRESS1]: 123 Main St, New York, NY 10001
                [PHONE1]: (555) 123-4567

                Masked text:
                [PERSON1] lives at [ADDRESS1]. His phone number is [PHONE1].

                Example 2:
                Original text:
                Sarah Johnson ordered a laptop from TechStore on 2023-05-15. Her email is sarah.j@email.com.

                Placeholder list:
                [PERSON1]: Sarah Johnson
                [PRODUCT1]: laptop
                [STORE1]: TechStore
                [DATE1]: 2023-05-15
                [EMAIL1]: sarah.j@email.com

                Masked text:
                [PERSON1] ordered a [PRODUCT1] from [STORE1] on [DATE1]. Her email is [EMAIL1].

                Example 3:
                Original text:
                Dr. Emily Brown, born on 1985-03-22, works at Central Hospital. Her patient, Mr. David Lee, has an appointment on 2023-06-10 at 2:30 PM.

                Placeholder list:
                [PERSON1]: Dr. Emily Brown
                [DATE1]: 1985-03-22
                [HOSPITAL1]: Central Hospital
                [PERSON2]: Mr. David Lee
                [DATE2]: 2023-06-10
                [TIME1]: 2:30 PM

                Masked text:
                [PERSON1], born on [DATE1], works at [HOSPITAL1]. Her patient, [PERSON2], has an appointment on [DATE2] at [TIME1].

                Now, please mask the following text:

                Text to mask:
                {content}

                Give me the placeholder list with the value and respective placeholder, and then the Masked text with the placeholders.
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
                Convert the following masked texts and placeholder lists into a JSON format. For each example, the JSON should have two main keys: "mapping" (a dictionary of placeholders and their original values) and "masked_text" (the text with placeholders).
                Always use [], not "" or ''
                You can only have one placeholder for each type and vice versa.
                Make sure that masked_text contains no sensitive information, only the placeholders.
                Keep all values, doesnt constitute sensitive information
                
                Example 1:
                Placeholder list:
                [PERSON1]: John Smith
                [ADDRESS1]: 123 Main St, New York, NY 10001
                [PHONE1]: (555) 123-4567

                Masked text:
                [PERSON1] lives at [ADDRESS1]. His phone number is [PHONE1].

                Output:
                {{
                    "mapping": {{
                        "[PERSON1]": "John Smith",
                        "[ADDRESS1]": "123 Main St, New York, NY 10001",
                        "[PHONE1]": "(555) 123-4567"
                    }},
                    "masked_text": "[PERSON1] lives at [ADDRESS1]. His phone number is [PHONE1]."
                }}

                Example 2:
                Placeholder list:
                [PERSON1]: Sarah Johnson
                [PRODUCT1]: laptop
                [STORE1]: TechStore
                [DATE1]: 2023-05-15
                [EMAIL1]: sarah.j@email.com

                Masked text:
                [PERSON1] ordered a [PRODUCT1] from [STORE1] on [DATE1]. Her email is [EMAIL1].

                Output:
                {{
                    "mapping": {{
                        "[PERSON1]": "Sarah Johnson",
                        "[PRODUCT1]": "laptop",
                        "[STORE1]": "TechStore",
                        "[DATE1]": "2023-05-15",
                        "[EMAIL1]": "sarah.j@email.com"
                    }},
                    "masked_text": "[PERSON1] ordered a [PRODUCT1] from [STORE1] on [DATE1]. Her email is [EMAIL1]."
                }}

                Example 3:
                Placeholder list:
                [PERSON1]: Dr. Emily Brown
                [DATE1]: 1985-03-22
                [HOSPITAL1]: Central Hospital
                [PERSON2]: Mr. David Lee
                [DATE2]: 2023-06-10
                [TIME1]: 2:30 PM

                Masked text:
                [PERSON1], born on [DATE1], works at [HOSPITAL1]. Her patient, [PERSON2], has an appointment on [DATE2] at [TIME1].

                Output:
                {{
                    "mapping": {{
                        "[PERSON1]": "Dr. Emily Brown",
                        "[DATE1]": "1985-03-22",
                        "[HOSPITAL1]": "Central Hospital",
                        "[PERSON2]": "Mr. David Lee",
                        "[DATE2]": "2023-06-10",
                        "[TIME1]": "2:30 PM"
                    }},
                    "masked_text": "[PERSON1], born on [DATE1], works at [HOSPITAL1]. Her patient, [PERSON2], has an appointment on [DATE2] at [TIME1]."
                }}

                Example 4:
                Placeholder list:
                [COMPANY1]: Company XYZ
                [PERSON1]: Jane Doe
                [COMPANY2]: ABC Corp
                [DATE1]: July 1, 2023
                [AMOUNT1]: $500 million

                Masked text:
                [COMPANY1]'s CEO, [PERSON1], announced a merger with [COMPANY2] on [DATE1]. The deal is valued at [AMOUNT1].

                Output:
                {{
                    "mapping": {{
                        "[COMPANY1]": "Company XYZ",
                        "[PERSON1]": "Jane Doe",
                        "[COMPANY2]": "ABC Corp",
                        "[DATE1]": "July 1, 2023",
                        "[AMOUNT1]": "$500 million"
                    }},
                    "masked_text": "[COMPANY1]'s CEO, [PERSON1], announced a merger with [COMPANY2] on [DATE1]. The deal is valued at [AMOUNT1]."
                }}

                Example 5:
                Placeholder list:
                [CREDITCARD1]: 4111-1111-1111-1111
                [PERSON1]: Michael Johnson
                [DATE1]: 12/25
                [CVV1]: 123

                Masked text:
                The credit card number [CREDITCARD1] belongs to [PERSON1], expiring on [DATE1], with CVV [CVV1].

                Output:
                {{
                    "mapping": {{
                        "[CREDITCARD1]": "4111-1111-1111-1111",
                        "[PERSON1]": "Michael Johnson",
                        "[DATE1]": "12/25",
                        "[CVV1]": "123"
                    }},
                    "masked_text": "The credit card number [CREDITCARD1] belongs to [PERSON1], expiring on [DATE1], with CVV [CVV1]."
                }}

                Now, please convert the following masked text and placeholder list into JSON format:

                {response_step1_content}

                ##JSON
                """
            }
        ]

        response_step2 = self.llm.request(messages_step2, MaskContract)

        masked_text = response_step2.masked_text
        mapping = response_step2.mapping

        for placeholder, value in mapping.items():
            if value in masked_text:
                masked_text = masked_text.replace(value, placeholder)

        response_step2.masked_text = masked_text

        return response_step2

    def get_placeholder(self, info_type):
        if info_type not in self.placeholder_counter:
            self.placeholder_counter[info_type] = 0
        self.placeholder_counter[info_type] += 1
        return f"[{info_type}{self.placeholder_counter[info_type]}]"

    def unmask_content(self, masked_content: str, mapping: dict) -> str:
        for placeholder, original in mapping.items():
            masked_content = masked_content.replace(placeholder, original)
        return masked_content