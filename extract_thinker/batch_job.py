import asyncio
from typing import Any, List, Type, Iterator, Optional
from pydantic import BaseModel
from openai import OpenAI
from instructor.batch import BatchJob as InstructorBatchJob
import json
import os

SLEEP_TIME = 60

class BatchJob:
    def __init__(
        self,
        messages_batch: Iterator[List[dict]],
        model: str,
        response_model: Type[BaseModel],
        file_path: str,
        output_path: str,
        api_key: str = os.getenv("OPENAI_API_KEY")
    ):
        self.response_model = response_model
        self.output_path = output_path
        self.file_path = file_path
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.batch_id = None
        self.file_id = None

        # Create the batch job input file (.jsonl)
        InstructorBatchJob.create_from_messages(
            messages_batch=messages_batch,
            model=model,
            file_path=file_path,
            response_model=response_model
        )

        self._add_method_to_file()

        # Upload file and create batch job
        self.file_id = self._upload_file()
        if not self.file_id:
            raise ValueError("Failed to upload file")

        self.batch_id = self._create_batch_job()
        if not self.batch_id:
            raise ValueError("Failed to create batch job")
        
    def _add_method_to_file(self) -> None:
        """Transform the JSONL file to match OpenAI's batch request format."""
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        
        with open(self.file_path, 'w') as file:
            for line in lines:
                data = json.loads(line)

                new_data = {
                    "custom_id": data["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": data["params"]["model"],
                        "messages": data["params"]["messages"],
                        "max_tokens": data["params"]["max_tokens"],
                        "temperature": data["params"]["temperature"],
                        "tools": data["params"]["tools"],
                        "tool_choice": data["params"]["tool_choice"]
                    }
                }
                file.write(json.dumps(new_data) + '\n')

    def _upload_file(self) -> Optional[str]:
        """Upload the JSONL file to OpenAI."""
        try:
            with open(self.file_path, "rb") as file:
                response = self.client.files.create(
                    file=file,
                    purpose="batch"
                )
                return response.id
        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    def _create_batch_job(self) -> Optional[str]:
        """Create a batch job via OpenAI API."""
        try:
            batch = self.client.batches.create(
                input_file_id=self.file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            return batch.id
        except Exception as e:
            print(f"Error creating batch job: {e}")
            return None

    async def get_status(self) -> str:
        """
        Get the current status of the batch job.
        Returns: queued, processing, completed, or failed
        """
        try:
            batch = await asyncio.to_thread(
                self.client.batches.retrieve,
                self.batch_id
            )
            return self._map_status(batch.status)
        except Exception as e:
            print(f"Error getting batch status: {e}")
            return "failed"

    def _map_status(self, api_status: str) -> str:
        """Maps OpenAI API status to simplified status."""
        status_mapping = {
            'validating': 'queued',
            'in_progress': 'processing',
            'finalizing': 'processing',
            'completed': 'completed',
            'failed': 'failed',
            'expired': 'failed',
            'cancelling': 'processing',
            'cancelled': 'failed'
        }
        return status_mapping.get(api_status, 'failed')

    async def get_result(self) -> BaseModel:
        """
        Wait for job completion and return parsed results using Instructor.
        Returns a tuple of (parsed_results, unparsed_results).
        
        parsed_results: List of successfully parsed objects matching response_model
        unparsed_results: List of results that failed to parse
        """
        try:
            # Wait until the batch is complete
            while True:
                status = await self.get_status()
                if status == 'completed':
                    break
                elif status == 'failed':
                    raise ValueError("Batch job failed")
                await asyncio.sleep(SLEEP_TIME)

            # Get batch details
            batch = await asyncio.to_thread(
                self.client.batches.retrieve,
                self.batch_id
            )
            
            if not batch.output_file_id:
                raise ValueError("No output file ID found")

            # Download the output file
            response = await asyncio.to_thread(
                self.client.files.content,
                batch.output_file_id
            )
            
            # Save the output file
            with open(self.output_path, 'w') as f:
                f.write(response.text)

            # Use Instructor to parse the results
            parsed, unparsed = InstructorBatchJob.parse_from_file(
                file_path=self.output_path,
                response_model=self.response_model
            )
            
            return parsed[0]
                
        except Exception as e:
            raise ValueError(f"Failed to process output file: {e}")
        finally:
            self._cleanup_files()

    async def cancel(self) -> bool:
        """Cancel the current batch job and confirm cancellation."""
        if not self.batch_id:
            print("No batch job to cancel.")
            return False
        
        try:
            await asyncio.to_thread(
                self.client.batches.cancel,
                self.batch_id
            )
            print("Batch job canceled successfully.")
            self._cleanup_files()
            return True
        except Exception as e:
            print(f"Error cancelling batch: {e}")
            return False

    def _cleanup_files(self):
        """Remove temporary files and batch directory if empty"""
        try:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
            
            # Try to remove parent directory if empty
            batch_dir = os.path.dirname(self.file_path)
            if os.path.exists(batch_dir) and not os.listdir(batch_dir):
                os.rmdir(batch_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup batch files: {e}")

    def __del__(self):
        """Cleanup files when object is destroyed"""
        self._cleanup_files()