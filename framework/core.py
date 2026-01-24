import json
import time
import concurrent.futures
import threading
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from openai import OpenAI
from .utils import Logger, format_time, CodeExecutor
from .config import ConfigManager
from .rate_limiter import RateLimiter

class EvalConfig:
    def __init__(self):
        config_manager = ConfigManager()
        model_config = config_manager.get_selected_model_config()
        
        self.api_key = model_config.get("api_key")
        self.base_url = model_config.get("base_url")
        self.model_name = model_config.get("model_name")
        self.temperature = model_config.get("temperature")
        self.rpm_limit = model_config.get("rpm_limit")
        self.tpm_limit = model_config.get("tpm_limit")
        
        self.pass_k = config_manager.get_global_setting("pass_k", 1)
        self.max_workers = config_manager.get_global_setting("workers", 1)
        
        self.input_cost_per_m = model_config.get("input_cost_per_m", 0.0)
        self.output_cost_per_m = model_config.get("output_cost_per_m", 0.0)

class LLMClient:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.max_retries = 3
        self.rate_limiter = RateLimiter(rpm_limit=config.rpm_limit, tpm_limit=config.tpm_limit)

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 4096) -> Tuple[str, Dict[str, int], Optional[str]]:
        estimated_input_chars = sum(len(m["content"]) for m in messages)
        estimated_tokens = int(estimated_input_chars * 0.25) + max_tokens
        
        self.rate_limiter.acquire(tokens=estimated_tokens)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.config.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens
                }
                
                if self.config.temperature is not None:
                    kwargs["temperature"] = self.config.temperature

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content:
                    continue
                
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
                return content.strip(), usage, None
            except Exception as e:
                error_str = str(e)
                # Regex to handle both single and double quoted string representations of the dict
                match = re.search(r"Error code: (\d+).*?'message':\s*(['\"])(.*?)\2", error_str, re.DOTALL)
                if match:
                    code = match.group(1)
                    message = match.group(3)
                    last_error = f"Error code: {code} - {message}"
                else:
                    # Fallback
                    last_error = f"Error: {error_str}".replace("\n", " ")
                time.sleep(1)
        
        return "", {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}, last_error

class BaseTask(ABC):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_data(self) -> List[Any]:
        problems = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    if "task_id" not in data and "id" not in data:
                        data["_index"] = i
                    problems.append(data)
        return problems

    @abstractmethod
    def process_item(self, item: Any, llm_client: LLMClient) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def log_columns(self) -> List[str]:
        pass

class CodeGenerationTask(BaseTask):
    """Base class for tasks involving code generation and execution."""
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.code_executor = CodeExecutor()

    @property
    def log_columns(self) -> List[str]:
        return ["task_id", "status", "duration", "tokens"]

    def _execute_and_log(self, task_id: str, code: str, start_time: float, usage: Dict[str, int]) -> Dict[str, Any]:
        status, _ = self.code_executor.execute(code)
        duration = time.time() - start_time
        return {
            "task_id": task_id,
            "status": status,
            "duration": format_time(duration),
            "duration_raw": duration,
            "tokens": usage.get("total_tokens", 0)
        }

class ReasoningTask(BaseTask):
    """Base class for reasoning tasks (e.g., GSM8K)."""
    @property
    def log_columns(self) -> List[str]:
        return ["id", "status", "ground_truth", "model_prediction", "duration", "tokens"]

class Runner:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.llm_client = LLMClient(config)
        self.stop_event = threading.Event()
        self.api_failure_occurred = False

    def run(self, task: BaseTask, task_name: str):
        problems = task.load_data()
        if not problems:
            print(f"No task data loaded. Please check the file: {task.dataset_path}")
            return

        print(f"Starting evaluation for {task_name}, Total tasks: {len(problems)} (Workers: {self.config.max_workers})...")
        print(f"Model: {self.config.model_name}")
        
        start_time_wall = time.time()
        passed_count = 0
        failed_count = 0
        api_failed_count = 0
        total_duration = 0
        total_tokens = 0
        
        with Logger(self.config.model_name, task_name) as logger:
            logger.write_header(task.log_columns)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for item in problems:
                    if self.stop_event.is_set():
                        break
                    future = executor.submit(self._safe_process_item, task, item, logger)
                    futures.append(future)
                
                with tqdm(total=len(futures), desc="Progress") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        if self.stop_event.is_set():
                            for f in futures:
                                f.cancel()
                            break
                            
                        try:
                            result = future.result()
                            if not result:
                                continue

                            logger.log_result(result, task.log_columns)
                            
                            status = result.get("status")
                            
                            if status == "CRITICAL_API_FAILURE":
                                api_failed_count += 1
                                self.api_failure_occurred = True
                                self.stop_event.set()
                                error_msg = result.get("error_msg", "Unknown API Error")
                                logger.log_message(f"CRITICAL API FAILURE: {error_msg}", level="ERROR")
                                print(f"\n[FATAL] Stopping execution due to API Error: {error_msg}")
                                break

                            elif status == "PASSED":
                                passed_count += 1
                            elif status == "API_FAILED":
                                api_failed_count += 1
                                logger.log_message(f"API Failed for item {result.get('id')}: {result.get('error_msg')}", level="WARN")
                            else:
                                failed_count += 1
                            
                            total_duration += result.get("duration_raw", 0)
                            total_tokens += result.get("tokens", 0)
                            
                        except Exception as e:
                            logger.log_message(f"Runner Loop Exception: {e}", level="ERROR")
                        finally:
                            pbar.update(1)
            
            end_time_wall = time.time()
            wall_time = end_time_wall - start_time_wall
            
            if self.api_failure_occurred:
                logger.log_summary("\n[!] Evaluation Aborted due to Critical API Failure.")
            
            self._print_summary(logger, passed_count, failed_count, api_failed_count, len(problems), 
                              total_duration, total_tokens, wall_time)

    def _safe_process_item(self, task: BaseTask, item: Any, logger: Logger) -> Optional[Dict[str, Any]]:
        if self.stop_event.is_set():
            return None
        
        try:
            return task.process_item(item, self.llm_client)
        except Exception as e:
            logger.log_message(f"Exception processing item: {e}", level="ERROR")
            return {
                "id": "unknown",
                "status": "INTERNAL_ERROR",
                "error_msg": str(e),
                "duration_raw": 0,
                "tokens": 0
            }

    def _print_summary(self, logger, passed, failed, api_errors, total, total_duration, total_tokens, wall_time):
        avg_duration = total_duration / total if total > 0 else 0
        avg_tokens = total_tokens / total if total > 0 else 0
        
        processed_total = passed + failed + api_errors
        accuracy = passed / processed_total if processed_total > 0 else 0.0
        tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0.0
        
        summary = (
            f"\n{'=' * 50}\n"
            f"Evaluation Summary\n"
            f"{'-' * 20}\n"
            f"Tasks Total: {total}\n"
            f"Tasks Processed: {processed_total}\n"
            f"Passed: {passed}\n"
            f"Failed: {failed}\n"
            f"API Errors: {api_errors}\n"
            f"Accuracy: {accuracy:.2%}\n"
            f"\nPerformance Metrics\n"
            f"{'-' * 20}\n"
            f"Wall Clock Time: {format_time(wall_time)}\n"
            f"Throughput: {tokens_per_sec:.1f} tokens/sec\n"
            f"Total Tokens: {total_tokens}\n"
            f"\nResults saved to: {logger.session_dir}\n"
            f"{'=' * 50}\n"
        )
        logger.log_summary(summary)
