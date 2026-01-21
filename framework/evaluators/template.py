import json
import time
from typing import List, Dict, Any
from ..core import BaseTask, LLMClient
from ..utils import format_time
from ..registry import TaskRegistry

# 1. Register your task name here
# This name must match the 'task' field in config.yaml
@TaskRegistry.register("my_new_task")
class TemplateTask(BaseTask):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    # 2. Implement data loading
    def load_data(self) -> List[Any]:
        """
        Load data from the dataset file.
        Returns a list of items (dict, str, etc.) to be processed.
        """
        problems = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        return problems

    # 3. Define log columns
    @property
    def log_columns(self) -> List[str]:
        """
        Define the columns for the result log file.
        Must include 'status', 'duration', 'tokens'.
        """
        return ["id", "status", "question", "answer", "duration", "tokens"]

    # 4. Implement item processing logic
    def process_item(self, item: Any, llm_client: LLMClient) -> Dict[str, Any]:
        """
        Process a single item using the LLM client.
        """
        # Extract data from item
        question = item.get("question", "")
        task_id = item.get("id", "unknown")
        
        # Construct prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
        
        # Call LLM
        start_time = time.time()
        completion, usage = llm_client.generate(messages)
        duration = time.time() - start_time
        
        # Check result
        if not completion:
            return {
                "id": task_id,
                "status": "API_FAILED",
                "duration": format_time(duration),
                "duration_raw": duration,
                "tokens": 0
            }

        # Validate answer (Implement your own logic here)
        # For example, just checking if the completion is non-empty
        is_passed = len(completion) > 0
        
        return {
            "id": task_id,
            "status": "PASSED" if is_passed else "FAILED",
            "question": question,
            "answer": completion,
            "duration": format_time(duration),
            "duration_raw": duration,
            "tokens": usage.get("total_tokens", 0)
        }
