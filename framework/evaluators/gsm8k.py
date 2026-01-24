import json
import time
import re
from typing import List, Dict, Any
from ..core import ReasoningTask, LLMClient
from ..utils import format_time
from ..registry import TaskRegistry

@TaskRegistry.register("gsm")
class GSM8KTask(ReasoningTask):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def process_item(self, item: Any, llm_client: LLMClient) -> Dict[str, Any]:
        index = item["_index"]
        question = item["question"]
        ground_truth_text = item["answer"]
        
        ground_truth_val = self._extract_answer(ground_truth_text)
        
        user_prompt = f"Question: {question}\nPlease reason step by step, and put your final answer at the end in the format: #### <answer>\nFor example: #### 1234"
        
        messages = [{"role": "user", "content": user_prompt}]
        
        start_time = time.time()
        completion, usage, error_msg = llm_client.generate(messages)
        duration = time.time() - start_time
        
        if error_msg:
            return {
                "id": index,
                "status": "CRITICAL_API_FAILURE",
                "error_msg": error_msg,
                "ground_truth": ground_truth_val,
                "model_prediction": "Error",
                "duration": format_time(duration),
                "duration_raw": duration,
                "tokens": 0
            }

        if not completion:
            return {
                "id": index,
                "status": "CRITICAL_API_FAILURE", # Changed from API_FAILED to CRITICAL_API_FAILURE
                "error_msg": "Empty completion (Possible Content Filter or Overload)",
                "ground_truth": ground_truth_val,
                "model_prediction": "None",
                "duration": format_time(duration),
                "duration_raw": duration,
                "tokens": 0
            }

        model_val = self._extract_answer(completion)
        is_correct = self._is_correct(model_val, ground_truth_val)
        
        return {
            "id": index,
            "status": "PASSED" if is_correct else "FAILED",
            "ground_truth": ground_truth_val,
            "model_prediction": model_val,
            "duration": format_time(duration),
            "duration_raw": duration,
            "tokens": usage.get("total_tokens", 0)
        }

    def _extract_answer(self, text: str) -> str:
        if not text:
            return None
        match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')
        matches = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
        if matches:
            return matches[-1].replace(',', '')
        return None

    def _is_correct(self, model_answer: str, ground_truth: str) -> bool:
        if model_answer is None or ground_truth is None:
            return False
        try:
            return float(model_answer) == float(ground_truth)
        except ValueError:
            return str(model_answer).strip() == str(ground_truth).strip()
