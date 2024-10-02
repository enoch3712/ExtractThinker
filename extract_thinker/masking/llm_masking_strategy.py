from extract_thinker.masking.abstract_masking_strategy import AbstractMaskingStrategy
from extract_thinker.models.MaskContract import MaskContract
from extract_thinker.llm import LLM

class LLMMaskingStrategy(AbstractMaskingStrategy):
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
                Make sure that masked_text contains no sensitive information, only the placeholders.

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

                Now, please convert the following masked text and placeholder list into JSON format:

                {response_step1_content}

                ##JSON
                """
            }
        ]

        return self.llm.request(messages_step2, MaskContract)

    def unmask_content(self, masked_content: str, mapping: dict) -> str:
        for placeholder, original in mapping.items():
            masked_content = masked_content.replace(placeholder, original)
        return masked_content