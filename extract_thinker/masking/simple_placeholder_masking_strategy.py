import re
from extract_thinker.llm import LLM
from extract_thinker.masking.abstract_masking_strategy import AbstractMaskingStrategy
from extract_thinker.models.MaskContract import MaskContract


class SimplePlaceholderMaskingStrategy(AbstractMaskingStrategy):
    MASK_PII_PROMPT = (
        "You are an AI assistant that masks only Personally Identifiable Information (PII) in text. "
        "Replace PII with placeholders in the format [TYPE#], e.g., [PERSON1], [ADDRESS1], [EMAIL1], etc. "
        "Do not mask numerical values or non-PII data. Ensure placeholders do not contain underscores or spaces."
    )

    MASK_PII_USER_PROMPT = """Please mask all PII in the following text. Replace PII with placeholders like [PERSON1], [ADDRESS1], [EMAIL1], etc. Do not use underscores or spaces in placeholders. Do not mask numerical values, dates, amounts, or other non-PII information. Return the masked text and a list of placeholders with their original values.

Here are some examples:

Example 1:
Original text:
John Smith lives at 123 Main St, New York, NY 10001. His phone number is (555) 123-4567. He deposited $5,000 on 2023-07-15.

Placeholder list:
[PERSON1]: John Smith
[ADDRESS1]: 123 Main St, New York, NY 10001
[PHONE1]: (555) 123-4567

Masked text:
[PERSON1] lives at [ADDRESS1]. His phone number is [PHONE1]. He deposited $5,000 on 2023-07-15.

Example 2:
Original text:
Sarah Johnson ordered a laptop from TechStore on 2023-05-15. Her email is sarah.j@email.com. The total amount was $1,200.

Placeholder list:
[PERSON1]: Sarah Johnson
[EMAIL1]: sarah.j@email.com

Masked text:
[PERSON1] ordered a laptop from TechStore on 2023-05-15. Her email is [EMAIL1]. The total amount was $1,200.

Example 3:
Original text:
Dr. Emily Brown, born on 1985-03-22, works at Central Hospital. Her patient, Mr. David Lee, has an appointment on 2023-06-10 at 2:30 PM. The procedure costs $3,500.

Placeholder list:
[PERSON1]: Dr. Emily Brown
[PERSON2]: Mr. David Lee

Masked text:
[PERSON1], born on 1985-03-22, works at Central Hospital. Her patient, [PERSON2], has an appointment on 2023-06-10 at 2:30 PM. The procedure costs $3,500.

Now, please mask the following text:

Text to mask:
{content}

Provide the placeholder list with their original values, followed by the masked text.
"""

    CONVERT_TO_JSON_PROMPT = (
        "You are an AI assistant that converts masked text information into JSON format, "
        "preserving only the masking for PII. Ensure that placeholders are strictly in the format [TYPE#], "
        "without underscores or spaces."
    )

    CONVERT_TO_JSON_USER_PROMPT = """Convert the following masked texts and placeholder lists into a JSON format. For each example, the JSON should have two main keys: "mapping" (a dictionary of placeholders and their original PII values) and "masked_text" (the text with PII placeholders). Do not include non-PII information such as numerical values, dates, or amounts. Ensure placeholders are in the correct format [TYPE#], without underscores or spaces.

Example 1:
Placeholder list:
[PERSON1]: John Smith
[ADDRESS1]: 123 Main St, New York, NY 10001
[PHONE1]: (555) 123-4567

Masked text:
[PERSON1] lives at [ADDRESS1]. His phone number is [PHONE1]. He deposited $5,000 on 2023-07-15.

Output:
{{
    "mapping": {{
        "[PERSON1]": "John Smith",
        "[ADDRESS1]": "123 Main St, New York, NY 10001",
        "[PHONE1]": "(555) 123-4567"
    }},
    "masked_text": "[PERSON1] lives at [ADDRESS1]. His phone number is [PHONE1]. He deposited $5,000 on 2023-07-15."
}}

Example 2:
Placeholder list:
[PERSON1]: Sarah Johnson
[EMAIL1]: sarah.j@email.com

Masked text:
[PERSON1] ordered a laptop from TechStore on 2023-05-15. Her email is [EMAIL1]. The total amount was $1,200.

Output:
{{
    "mapping": {{
        "[PERSON1]": "Sarah Johnson",
        "[EMAIL1]": "sarah.j@email.com"
    }},
    "masked_text": "[PERSON1] ordered a laptop from TechStore on 2023-05-15. Her email is [EMAIL1]. The total amount was $1,200."
}}

Example 3:
Placeholder list:
[PERSON1]: Dr. Emily Brown
[PERSON2]: Mr. David Lee

Masked text:
[PERSON1], born on 1985-03-22, works at Central Hospital. Her patient, [PERSON2], has an appointment on 2023-06-10 at 2:30 PM. The procedure costs $3,500.

Output:
{{
    "mapping": {{
        "[PERSON1]": "Dr. Emily Brown",
        "[PERSON2]": "Mr. David Lee"
    }},
    "masked_text": "[PERSON1], born on 1985-03-22, works at Central Hospital. Her patient, [PERSON2], has an appointment on 2023-06-10 at 2:30 PM. The procedure costs $3,500."
}}

Now, please convert the following masked text and placeholder list into JSON format:

{response_step1_content}

##JSON
"""

    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.placeholder_counter = {}

    async def mask_content(self, content: str) -> MaskContract:
        response_step1_content = await self._step1_mask_pii(content)
        mask_contract = await self._step2_convert_to_json(response_step1_content)
        self._validate_placeholders(mask_contract)
        return mask_contract

    async def _step1_mask_pii(self, content: str) -> str:
        messages_step1 = [
            {"role": "system", "content": self.MASK_PII_PROMPT},
            {"role": "user", "content": self.MASK_PII_USER_PROMPT.format(content=content)},
        ]
        response_step1 = self.llm.request(messages_step1)
        response_step1_content = response_step1.choices[0].message.content
        return response_step1_content

    async def _step2_convert_to_json(self, response_step1_content: str) -> MaskContract:
        messages_step2 = [
            {"role": "system", "content": self.CONVERT_TO_JSON_PROMPT},
            {
                "role": "user",
                "content": self.CONVERT_TO_JSON_USER_PROMPT.format(
                    response_step1_content=response_step1_content
                ),
            },
        ]
        response_step2 = self.llm.request(messages_step2, MaskContract)
        masked_text = response_step2.masked_text
        mapping = response_step2.mapping

        for placeholder, value in mapping.items():
            if value in masked_text:
                masked_text = masked_text.replace(value, placeholder)
        response_step2.masked_text = masked_text
        return response_step2

    def _validate_placeholders(self, mask_contract: MaskContract):
        placeholder_pattern = re.compile(r'^\[[A-Za-z_]+[0-9]*\]$')
        for placeholder in mask_contract.mapping.keys():
            if not placeholder_pattern.match(placeholder):
                raise ValueError(f"Invalid placeholder format: {placeholder}")

    def unmask_content(self, masked_content: str, mapping: dict) -> str:
        for placeholder, original in mapping.items():
            masked_content = masked_content.replace(placeholder, original)
        return masked_content