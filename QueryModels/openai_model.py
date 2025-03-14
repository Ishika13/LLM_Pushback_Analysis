import os
import json
import time
import tempfile
import openai

class OpenAIBatchModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = openai.OpenAI()  # Instantiates the client using the new SDK

    def predict_withformatting(self, queries, formatting, batch_size=32, max_tokens=500):
        # Build a temporary JSONL file with one request per query.
        lines = []
        for i, query in enumerate(queries):
            custom_id = f"request-{i}"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ],
                "max_tokens": max_tokens
            }
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": payload
            }
            lines.append(line)

        temp_file_path = tempfile.mktemp(suffix=".jsonl")
        with open(temp_file_path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        print(f"Wrote {len(lines)} lines to {temp_file_path}")

        # Upload the JSONL file using the new client interface.
        with open(temp_file_path, "rb") as f:
            uploaded_file = self.client.files.create(file=f, purpose="batch")
        print(f"Uploaded file {uploaded_file.id}")
        file_id = uploaded_file.id
        
        # Create the batch job.
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_id = batch.id
        print(f"Created batch job {batch_id}")
        
        # Poll until the batch completes (with a timeout).
        timeout = 600  # 10 minutes
        poll_interval = 10
        start_time = time.time()
        while True:
            current_batch = self.client.batches.retrieve(batch_id)
            print(f"Batch status: {current_batch.status}")
            if current_batch.status == "completed":
                break
            elif current_batch.status in ["failed", "expired"]:
                raise Exception(f"Batch job {batch_id} failed with status {current_batch.status}")
            if time.time() - start_time > timeout:
                raise Exception("Batch job timed out")
            time.sleep(poll_interval)
        print(f"Current batch: {current_batch}")
        
        # Retrieve the output file using the new content method.
        output_file_id = current_batch.output_file_id
        file_response = self.client.files.content(output_file_id)
        output_content = file_response.text
        
        # Parse the output content into responses.
        responses = {}
        for line in output_content.splitlines():
            data = json.loads(line)
            custom_id = data.get("custom_id")
            if data.get("response") and data["response"].get("body"):
                choices = data["response"]["body"].get("choices", [])
                if choices:
                    responses[custom_id] = choices[0]["message"]["content"]
        ordered_responses = [responses.get(f"request-{i}", "") for i in range(len(queries))]
        
        # Clean up temporary file.
        os.remove(temp_file_path)
        return ordered_responses
