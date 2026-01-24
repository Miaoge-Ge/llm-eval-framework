import json
import time
import re
from typing import List, Dict, Any
from ..core import CodeGenerationTask, LLMClient
from ..utils import format_time
from ..registry import TaskRegistry

@TaskRegistry.register("humaneval")
@TaskRegistry.register("humanevalplus")
class HumanEvalTask(CodeGenerationTask):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.header = "from typing import List, Dict, Tuple, Optional, Union, Any, Set, Deque\nimport math\nimport re\nimport sys\nimport heapq\nimport itertools\nimport collections\nimport functools\n"

    def process_item(self, item: Any, llm_client: LLMClient) -> Dict[str, Any]:
        task_id = item["task_id"]
        prompt = item["prompt"]
        test_code = item["test"]
        entry_point = item["entry_point"]
        
        start_time = time.time()
        
        system_prompt = "You are an expert Python programmer. Complete the function based on the provided signature and docstring. Output only the code inside ```python``` blocks."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        completion, usage, error_msg = llm_client.generate(messages)
        
        if error_msg:
            return {
                "task_id": task_id,
                "status": "CRITICAL_API_FAILURE",
                "error_msg": error_msg,
                "duration": format_time(time.time() - start_time),
                "duration_raw": time.time() - start_time,
                "tokens": 0
            }

        if not completion:
            return {
                "task_id": task_id,
                "status": "CRITICAL_API_FAILURE", # Changed from API_FAILED to CRITICAL_API_FAILURE
                "error_msg": "Empty completion (Possible Content Filter or Overload)",
                "duration": format_time(time.time() - start_time),
                "duration_raw": time.time() - start_time,
                "tokens": 0
            }

        code = self._extract_code(completion)
        
        full_code = f"{self.header}\n{code}\n\n{test_code}\n\ncheck({entry_point})"
        
        return self._execute_and_log(task_id, full_code, start_time, usage)

    def _extract_code(self, text: str) -> str:
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        if "def " in text:
            return text
        return text
