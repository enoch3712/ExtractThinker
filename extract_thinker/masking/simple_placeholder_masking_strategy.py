import re
from extract_thinker.llm import LLM
from extract_thinker.masking.abstract_masking_strategy import AbstractMaskingStrategy
from extract_thinker.models.MaskContract import MaskContract, MaskContractDict


class SimplePlaceholderMaskingStrategy(AbstractMaskingStrategy):
    MASK_PII_PROMPT = (
        "You are an AI assistant that masks only Personally Identifiable Information (PII) in text. "
        "Replace PII with placeholders in the format [TYPE#], e.g., [PERSON1], [ADDRESS1], [EMAIL1], etc. "
        "Do not mask numerical values or non-PII data. Ensure placeholders do not contain underscores or spaces."
        "Do not mask the key value result, they will be masked later."
        "Dont return masked text, only the placeholder list."
        "Provide a step-by-step reasoning when identifying PII."
    )

    MASK_PII_USER_PROMPT = """Please mask only Personally Identifiable Information (PII) in the following text. Replace PII with placeholders like [PERSON1], [ADDRESS1], [EMAIL1], [PHONE1], [TAXID1], etc. Do not use underscores or spaces in placeholders. Mask names, addresses, email addresses, phone numbers (including international formats), tax IDs, and other PII. Do not mask numerical values (except phone numbers and tax IDs), dates, amounts, or other non-PII information. Return the masked text and a list of placeholders with their original values.

Here are some examples of correct masking:

Example 1:
Original text:
John Smith lives at 123 Main St, New York, NY 10001. His phone number is (555) 123-4567 and his SSN is 123-45-6789. For international calls, use +1-555-987-6543. He deposited $5,000 on 2023-07-15.

Placeholder list:
[PERSON1]: John Smith
[ADDRESS1]: 123 Main St, New York, NY 10001
[PHONE1]: (555) 123-4567
[TAXID1]: 123-45-6789
[PHONE2]: +1-555-987-6543

Example 2:
Original text:
Sarah Johnson ordered a laptop from TechStore on 2023-05-15. Her email is sarah.j@email.com and her work number is 1-800-555-1234. The company's EIN is 12-3456789. The total amount was $1,200.

Placeholder list:
[PERSON1]: Sarah Johnson
[EMAIL1]: sarah.j@email.com
[PHONE1]: 1-800-555-1234
[TAXID1]: 12-3456789

Example 3 (Demonstrating what NOT to mask):
Original text:
The company's revenue was $10,000,000 last year. Project XYZ has a budget of $500,000 and is due on 2023-12-31. The office can accommodate 50 employees.

Note: In this example, no masking is performed because there is no PII present. Numerical values (except phone numbers and tax IDs), project names, and dates are not considered PII.

Now, please mask the following text:

Text to mask:
{content}

Provide your step-by-step reasoning, the placeholder list with their original values, and the masked text.
"""

    CONVERT_TO_JSON_PROMPT = (
        "You are an AI assistant that converts placeholder lists into JSON format. "
        "Ensure that placeholders are strictly in the format [TYPE#], without underscores or spaces."
    )

    CONVERT_TO_JSON_USER_PROMPT = """Convert the following placeholder lists into a JSON format. For each example, the JSON should have a single key: "mapping" (a dictionary of placeholders and their original PII values). Ensure placeholders are in the correct format [TYPE#], without underscores or spaces.

Example 1:
Placeholder list:
[PERSON1]: John Smith
[ADDRESS1]: 123 Main St, New York, NY 10001
[PHONE1]: (555) 123-4567

Output:
{{
    "mapping": {{
        "[PERSON1]": "John Smith",
        "[ADDRESS1]": "123 Main St, New York, NY 10001",
        "[PHONE1]": "(555) 123-4567"
    }}
}}

Example 2:
Placeholder list:
[PERSON1]: Sarah Johnson
[EMAIL1]: sarah.j@email.com

Output:
{{
    "mapping": {{
        "[PERSON1]": "Sarah Johnson",
        "[EMAIL1]": "sarah.j@email.com"
    }}
}}

Now, please convert the following placeholder list into JSON format:

{response_step1_content}

##JSON
"""

    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.placeholder_counter = {}

    async def mask_content(self, content: str) -> MaskContract:
        response_step1_content = await self._step1_mask_pii(content)
        response_step2_content = await self._step2_convert_to_json(response_step1_content)
        result = self._parse_mask_contract_dict(response_step2_content.mapping, content)
        return result
    
    def _parse_mask_contract_dict(self, mapping: dict, content: str) -> MaskContract:
        masked_text = content
        for placeholder, value in mapping.items():
            masked_text = masked_text.replace(value, placeholder)
        return MaskContract(masked_text=masked_text, mapping=mapping)

    async def _step1_mask_pii(self, content: str) -> str:
        messages_step1 = [
            {"role": "system", "content": self.MASK_PII_PROMPT},
            {"role": "user", "content": self.MASK_PII_USER_PROMPT.format(content=content)},
        ]
        response_step1 = self.llm.request(messages_step1)
        response_step1_content = response_step1.choices[0].message.content
        return response_step1_content

    async def _step2_convert_to_json(self, response_step1_content: str) -> MaskContractDict:
        messages_step2 = [
            {"role": "system", "content": self.CONVERT_TO_JSON_PROMPT},
            {
                "role": "user",
                "content": self.CONVERT_TO_JSON_USER_PROMPT.format(
                    response_step1_content=response_step1_content,
                ),
            },
        ]
        response_step2 = self.llm.request(messages_step2, MaskContractDict)
        return response_step2
    
    def _validate_placeholders(self, mask_contract: MaskContract):
        placeholder_pattern = re.compile(r'^\[[A-Za-z]+[0-9]*\]$')
        for placeholder in mask_contract.mapping.keys():
            if not placeholder_pattern.match(placeholder):
                raise ValueError(f"Invalid placeholder format: {placeholder}")

    def unmask_content(self, masked_content: str, mapping: dict) -> str:
        for placeholder, original in mapping.items():
            masked_content = masked_content.replace(placeholder, original)
        return masked_content
