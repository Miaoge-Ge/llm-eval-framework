import json
import time
import re
from typing import List, Dict, Any
from ..core import CodeGenerationTask, LLMClient
from ..utils import format_time
from ..registry import TaskRegistry

@TaskRegistry.register("mbpp")
class MBPPTask(CodeGenerationTask):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.header = "from typing import List, Dict, Tuple, Optional, Union, Any, Set, Deque\nimport math\nimport re\nimport sys\nimport heapq\nimport itertools\nimport collections\nimport functools\n"

    def process_item(self, item: Any, llm_client: LLMClient) -> Dict[str, Any]:
        task_id = str(item["task_id"])
        prompt = item["text"]
        test_list = item["test_list"]
        
        function_name_hint = ""
        if test_list and len(test_list) > 0:
            first_test = test_list[0]
            match = re.search(r"assert\s+(\w+)\(", first_test)
            if match:
                func_name = match.group(1)
                function_name_hint = f"\nImportant: The function name MUST be `{func_name}`."

        user_prompt = f"Task: {prompt}{function_name_hint}\n\nPlease write Python code to solve this task."
        system_prompt = "You are an expert Python programmer. Output only the code inside ```python``` blocks."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        start_time = time.time()
        completion, usage, error_msg = llm_client.generate(messages, max_tokens=2048)
        
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
        test_code_block = "\n".join(test_list)
        full_code = f"{self.header}\n{code}\n\n{test_code_block}"
        
        return self._execute_and_log(task_id, full_code, start_time, usage)

    def _extract_code(self, text: str) -> str:
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        return text
