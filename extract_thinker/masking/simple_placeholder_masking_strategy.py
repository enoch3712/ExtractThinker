import re
from extract_thinker.llm import LLM
from extract_thinker.masking.abstract_masking_strategy import AbstractMaskingStrategy
from extract_thinker.models.MaskContract import MaskContract
import asyncio

class SimplePlaceholderMaskingStrategy(AbstractMaskingStrategy):
    MASK_PII_PROMPT = (
        "You are an AI assistant that masks only Personally Identifiable Information (PII) in text. "
        "Replace PII with placeholders in the format [TYPE#], e.g., [PERSON1], [ADDRESS1], [EMAIL1], etc. "
        "Do not mask numerical values or non-PII data. Ensure placeholders do not contain underscores or spaces."
    )

    MASK_PII_USER_PROMPT = """You are an AI assistant that masks only Personally Identifiable Information (PII) in text. Replace PII with placeholders in the format [TYPE#], e.g., [PERSON1], [ADDRESS1], [EMAIL1], etc. Do not mask numerical values or non-PII data.

Example 1:
Input:
John Smith lives at 123 Main St, New York, NY 10001. His phone number is (555) 123-4567 and his SSN is 123-45-6789. For international calls, use +1-555-987-6543. He deposited $5,000 on 2023-07-15.

Output:
##PLACEHOLDER LIST:
[PERSON1]: John Smith
[ADDRESS1]: 123 Main St, New York, NY 10001
[PHONE1]: (555) 123-4567
[TAXID1]: 123-45-6789
[PHONE2]: +1-555-987-6543

Input: 
{content}

Output:
"""

    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.global_mapping = {}  # Final mapping of placeholders to PII values
        self.pii_to_placeholder = {}  # Mapping of PII values to placeholders

    async def mask_content(self, content: str) -> MaskContract:
        paragraphs = self._split_into_paragraphs(content)
        masked_paragraphs = []
        all_mappings = []

        for paragraph in paragraphs:
            placeholder_list = await self._process_paragraph(paragraph)
            mapping = self._parse_placeholder_list(placeholder_list)
            all_mappings.append(mapping)
            # Mask the paragraph using the mapping
            masked_paragraph = self._apply_masking(paragraph, mapping)
            masked_paragraphs.append(masked_paragraph)

        # After processing all paragraphs, reconcile the mappings
        self._reconcile_mappings(all_mappings)

        # Combine masked paragraphs back into the final masked text
        masked_text = '\n\n'.join(masked_paragraphs)

        result = MaskContract(masked_text=masked_text, mapping=self.global_mapping)
        return result

    def _split_into_paragraphs(self, text: str) -> list:
        # Split text into paragraphs based on various newline patterns
        paragraphs = re.split(r'\n{2,}|\r\n{2,}|\r{2,}', text.strip())
        # Further split paragraphs if they contain single newlines
        result = []
        for paragraph in paragraphs:
            sub_paragraphs = paragraph.split('\n')
            result.extend(sub_para.strip() for sub_para in sub_paragraphs if sub_para.strip())
        return result

    async def _process_paragraph(self, paragraph: str) -> str:
        async def single_run():
            MAX_SINGLE_RUN_RETRIES = 2  # Maximum number of retries per single run
            attempt = 0
            while attempt <= MAX_SINGLE_RUN_RETRIES:
                messages_step1 = [
                    {"role": "system", "content": self.MASK_PII_PROMPT},
                    {"role": "user", "content": self.MASK_PII_USER_PROMPT.format(content=paragraph)},
                ]
                response_step1 = self.llm.request(messages_step1)
                response_step1_content = response_step1.choices[0].message.content

                if "##PLACEHOLDER LIST:" not in response_step1_content:
                    # Retry with an explicit reminder
                    attempt += 1
                    reminder_message = (
                        "The previous response did not include '##PLACEHOLDER LIST:'. "
                        "Please make sure to include '##PLACEHOLDER LIST:' followed by the placeholder list."
                    )
                    messages_step1.append({"role": "assistant", "content": response_step1_content})
                    messages_step1.append({"role": "user", "content": reminder_message})
                    continue  # Retry the request
                else:
                    # Extract the placeholder list
                    split_result = response_step1_content.split("##PLACEHOLDER LIST:")
                    placeholder_list = split_result[1].strip()
                    return placeholder_list

            # If all retries fail, raise an error
            raise ValueError("Unable to obtain a valid '##PLACEHOLDER LIST:' after multiple attempts.")

        MAX_RETRIES = 10  # Maximum number of retries for inconsistent results
        retry_count = 0

        while retry_count < MAX_RETRIES:
            # Run two parallel requests
            initial_runs = [single_run() for _ in range(2)]
            initial_results = await asyncio.gather(*initial_runs)

            # Compare the initial two results
            if initial_results[0] != initial_results[1]:
                retry_count += 1
                continue  # Retry due to inconsistency

            # Reconcile all mappings
            all_results = initial_results
            final_placeholder_list = self._reconcile_placeholder_lists(all_results, paragraph)
            return final_placeholder_list

        # If all retries fail, raise an error
        raise ValueError("Unable to obtain consistent placeholder lists after maximum retries.")


    def _reconcile_placeholder_lists(self, placeholder_lists: list, paragraph: str) -> str:
        # Parse each placeholder list into a mapping
        mappings = [self._parse_placeholder_list(plist) for plist in placeholder_lists]

        # Collect counts of original values and their PII types
        original_value_counts = {}
        for mapping in mappings:
            for placeholder, original_value in mapping.items():
                # Extract PII type from placeholder, e.g., [PERSON1] -> PERSON
                m = re.match(r'\[([A-Za-z]+)[0-9]*\]', placeholder)
                if m:
                    pii_type = m.group(1)
                    key = (original_value, pii_type)
                    if key not in original_value_counts:
                        original_value_counts[key] = 0
                    original_value_counts[key] += 1

        # Keep original values that appear in all lists
        required_count = len(placeholder_lists)
        final_items = [
            (original_value, pii_type)
            for (original_value, pii_type), count in original_value_counts.items()
            if count == required_count
        ]

        # If no items appear consistently across all lists, accept all items
        if not final_items:
            for (original_value, pii_type), count in original_value_counts.items():
                final_items.append((original_value, pii_type))

        # Create a mapping for this paragraph without assigning placeholders yet
        paragraph_mapping = {}
        for original_value, pii_type in final_items:
            paragraph_mapping[original_value] = pii_type

        # Return the paragraph mapping as a placeholder list string
        placeholder_list = '\n'.join([f'{pii_type}: {original_value}' for original_value, pii_type in paragraph_mapping.items()])

        return placeholder_list

    def _parse_placeholder_list(self, placeholder_list_str: str) -> dict:
        mapping = {}
        lines = placeholder_list_str.strip().split('\n')
        for line in lines:
            if line.strip() == '':
                continue
            # Expected format: [PLACEHOLDER]: original_value
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue  # or raise an error
            placeholder = parts[0].strip()
            original_value = parts[1].strip()
            mapping[placeholder] = original_value
        return mapping

    def _reconcile_mappings(self, all_mappings: list):
        # Collect all PII values and their types
        pii_items = {}
        for mapping in all_mappings:
            for placeholder, original_value in mapping.items():
                m = re.match(r'\[([A-Za-z]+)[0-9]*\]', placeholder)
                if m:
                    pii_type = m.group(1)
                    if original_value not in pii_items:
                        pii_items[original_value] = pii_type
                    else:
                        # If the same PII value has different types, decide how to handle it
                        # For simplicity, we'll keep the first type encountered
                        pass

        # Assign placeholders to PII values
        placeholder_counters = {}
        for original_value, pii_type in pii_items.items():
            if pii_type not in placeholder_counters:
                placeholder_counters[pii_type] = 1
            else:
                placeholder_counters[pii_type] += 1
            placeholder = f'[{pii_type}{placeholder_counters[pii_type]}]'
            self.global_mapping[placeholder] = original_value
            self.pii_to_placeholder[original_value] = placeholder

    def _apply_masking(self, text: str, mapping: dict) -> str:
        masked_text = text
        # Use the global mapping to ensure consistency across paragraphs
        for original_value, pii_type in mapping.items():
            placeholder = self.pii_to_placeholder.get(original_value)
            if placeholder:
                masked_text = masked_text.replace(original_value, placeholder)
        return masked_text

    def unmask_content(self, masked_content: str, mapping: dict) -> str:
        # Sort placeholders by length to avoid partial replacements
        sorted_mapping = dict(sorted(mapping.items(), key=lambda x: -len(x[0])))
        for placeholder, original in sorted_mapping.items():
            masked_content = masked_content.replace(placeholder, original)
        return masked_content