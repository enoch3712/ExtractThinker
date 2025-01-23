import re
import hashlib
from extract_thinker.llm import LLM
from extract_thinker.masking.abstract_masking_strategy import AbstractMaskingStrategy
from extract_thinker.models.MaskContract import MaskContract, MaskContractDict
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64


class DeterministicHashingMaskingStrategy(AbstractMaskingStrategy):
    MASK_PII_PROMPT = (
        "You are an AI assistant that masks only Personally Identifiable Information (PII) in text. "
        "Replace PII with placeholders in the format [TYPE#], e.g., [PERSON1], [ADDRESS1], [EMAIL1], etc. "
        "Do not mask numerical values or non-PII data. Ensure placeholders do not contain underscores or spaces."
        "Do not mask the key value result, they will be masked later."
        "Don't return masked text, only the placeholder list."
        "Values and Amounts(e.g $1000) are not PII values. The same for dates"
        "Provide a step-by-step reasoning when identifying PII."
        "Always return ##Placeholder list: as part of the response"
    )

    MASK_PII_USER_PROMPT = """Task: Mask personally identifiable information (PII) in the provided text, replacing PII with placeholders like [PERSON1], [ADDRESS1], [EMAIL1], etc. Do not mask numerical values unless they are phone numbers or tax IDs. Return only the placeholder list with reasoning for each identified PII.

Step 1: Reasoning & Thought Process
1. Analyze the text:
   - Carefully examine each part of the text to determine if it contains PII.
   - Focus on identifying common types of PII such as names, email addresses, phone numbers, tax IDs, and physical addresses.
   - Ignore non-PII data such as dates, numerical values (except phone numbers and tax IDs), and any other non-sensitive information.

2. Justify the decision:
   - For each segment identified as PII, explain why it qualifies as such.
   - Clearly differentiate between PII and non-PII elements. Provide reasoning for why certain elements are not PII.

Step 2: Action
1. Mask PII:
   - Replace each identified PII with an appropriate placeholder in the format [TYPE#] (e.g., [PERSON1], [ADDRESS1]).
   - Do not mask any non-PII elements.

2. Return placeholder list:
   - Return a list of placeholders and their corresponding original values (but do not return the masked text).
   - Ensure placeholders are formatted without underscores or spaces.

Examples:

Example 1:
Original text:
John Smith lives at 123 Main St, New York, NY 10001. His phone number is (555) 123-4567 and his SSN is 123-45-6789. For international calls, use +1-555-987-6543. He deposited $5,000 on 2023-07-15.

Output:
##Placeholder list:
[PERSON1]: John Smith
[ADDRESS1]: 123 Main St, New York, NY 10001
[PHONE1]: (555) 123-4567
[TAXID1]: 123-45-6789
[PHONE2]: +1-555-987-6543

Example 2:
Original text:
Sarah Johnson ordered a laptop from TechStore on 2023-05-15. Her email is sarah.j@email.com and her work number is 1-800-555-1234. The company's EIN is 12-3456789. The total amount was $1,200.

Output:
##Placeholder list:
[PERSON1]: Sarah Johnson
[EMAIL1]: sarah.j@email.com
[PHONE1]: 1-800-555-1234
[TAXID1]: 12-3456789

Example 3 (Demonstrating what NOT to mask):
Original text:
The company's revenue was $10,000,000 last year. Project XYZ has a budget of $500,000 and is due on 2023-12-31. The office can accommodate 50 employees.

Note: In this example, no masking is performed because there is no PII present. Numerical values (except phone numbers and tax IDs), project names, and dates are not considered PII.

Example 4:
Original text:
John Doe transferred $5000 to Jane Smith on 2021-05-01.

Step 1: Reasoning & Thought Process
Upon analyzing the text "John Doe transferred $5000 to Jane Smith on 2021-05-01.", we need to identify any PII present.

1. Identifying PII Types: The common types of PII we're looking for are names (e.g., John Doe, Jane Smith), email addresses, phone numbers, tax IDs, and physical addresses.
2. Examining Text Segments:
   - "John Doe" - This is a name, which is a type of PII.
   - "Jane Smith" - This is another name, which is a type of PII.
   - "$5000" - This is a financial transaction amount, not a phone number or tax ID, so it's not a type of PII in this context. Numerical values like this are often found in everyday text and aren't PII.
   - "2021-05-01" - This is a date, which is not PII because it doesn't contain identifying information about a person.

Step 2: Action
Based on the identified PII types and segments, we'll create placeholders for each PII found.

1. Masking PII: We'll replace each identified PII with an appropriate placeholder in the format [TYPE#].
2. Returning Placeholder List: We'll return a list of placeholders and their corresponding original values.

Output:
##Placeholder list:
[PERSON1]: John Doe
[PERSON2]: Jane Smith

Text to mask:
{content}

Provide your step-by-step reasoning, and then return the placeholder list.
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
            hash_value = self._deterministic_hash(value)
            masked_text = masked_text.replace(value, f"{hash_value}")
        return MaskContract(masked_text=masked_text, mapping=mapping)

    async def _step1_mask_pii(self, content: str) -> str:
        messages_step1 = [
            {"role": "system", "content": self.MASK_PII_PROMPT},
            {"role": "user", "content": self.MASK_PII_USER_PROMPT.format(content=content)},
        ]
        response_step1 = self.llm.request(messages_step1)
        response_step1_content = response_step1.choices[0].message.content

        # Split the response into reasoning and the placeholder list
        split_result = response_step1_content.split("##Placeholder list:")
        if len(split_result) == 2:
            reasoning_part = split_result[0].strip()
            placeholder_list = split_result[1].strip()
        else:
            raise ValueError("Unexpected response format: 'Placeholder List' section not found.")

        # Return only the placeholder list
        return placeholder_list

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

    def _deterministic_hash(self, value: str) -> str:
        # Generate a deterministic hash using PBKDF2HMAC with SHA256
        salt = b'some_constant_salt'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        hashed = kdf.derive(value.encode())
        return base64.urlsafe_b64encode(hashed).decode('utf-8')
