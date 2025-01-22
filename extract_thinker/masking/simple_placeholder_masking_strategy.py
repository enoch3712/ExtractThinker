import re
from extract_thinker.llm import LLM
from extract_thinker.masking.abstract_masking_strategy import AbstractMaskingStrategy
from extract_thinker.models.MaskContract import MaskContract
import asyncio

class SimplePlaceholderMaskingStrategy(AbstractMaskingStrategy):
    MASK_PII_PROMPT = (
        "You are an AI assistant that masks only Personally Identifiable Information (PII) in text. "
        "Replace PII with placeholders in the format [TYPE#], e.g., [PERSON1], [ADDRESS1], [EMAIL1], etc. "
        "Do not mask numerical values or non-PII data. Values and Amounts(e.g $1000) are not PII values. "
        "The same applies for dates. Return the masked text and mapping in JSON format."
        "Rethink what you did to make sure that you masking every PII value."
    )

    MASK_PII_USER_PROMPT = '''Mask personally identifiable information (PII) in the provided text, replacing PII with placeholders like [PERSON1], [ADDRESS1], [EMAIL1], etc. Return both the masked text and mapping in JSON format.

Example:
Input:
John Smith lives at 123 Main St, New York, NY 10001. His phone number is (555) 123-4567 and his SSN is 123-45-6789. He deposited $5,000 on 2023-07-15.

Output:
{{
    "mapping": {{
        "[PERSON1]": "John Smith",
        "[ADDRESS1]": "123 Main St, New York, NY 10001",
        "[PHONE1]": "(555) 123-4567",
        "[TAXID1]": "123-45-6789"
    }},
    "masked_text": "[PERSON1] lives at [ADDRESS1]. His phone number is [PHONE1] and his SSN is [TAXID1]. He deposited $5,000 on 2023-07-15."
}}

Text to mask:
{content}

Return the response in JSON format.
##JSON'''

    def __init__(self, llm: LLM):
        super().__init__(llm)

    async def mask_content(self, content: str) -> MaskContract:
        messages = [
            {"role": "system", "content": self.MASK_PII_PROMPT},
            {"role": "user", "content": self.MASK_PII_USER_PROMPT.format(content=content)},
        ]
        
        return self.llm.request(messages, MaskContract)

    def unmask_content(self, masked_content: str, mapping: dict) -> str:
        unmasked_text = masked_content
        sorted_mapping = dict(sorted(mapping.items(), key=lambda x: -len(x[0])))
        for placeholder, original in sorted_mapping.items():
            unmasked_text = unmasked_text.replace(placeholder, original)
        return unmasked_text