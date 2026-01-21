import time
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from openai import OpenAI
from .utils import Logger, format_time
from .config import ConfigManager

class EvalConfig:
    def __init__(self):
        config_manager = ConfigManager()
        model_config = config_manager.get_selected_model_config()
        
        self.api_key = model_config.get("api_key")
        self.base_url = model_config.get("base_url")
        self.model_name = model_config.get("model_name")
        self.temperature = model_config.get("temperature")
        
        self.pass_k = config_manager.get_global_setting("pass_k", 1)
        self.max_workers = config_manager.get_global_setting("workers", 1)

class LLMClient:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.max_retries = 3

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 4096) -> Tuple[str, Dict[str, int]]:
        for _ in range(self.max_retries):
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
                return content.strip(), usage
            except Exception as e:
                time.sleep(1)
        return "", {"total_tokens": 0}

class BaseTask(ABC):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    @abstractmethod
    def load_data(self) -> List[Any]:
        pass

    @abstractmethod
    def process_item(self, item: Any, llm_client: LLMClient) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def log_columns(self) -> List[str]:
        pass

class Runner:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.llm_client = LLMClient(config)

    def run(self, task: BaseTask, task_name: str):
        problems = task.load_data()
        if not problems:
            print(f"No task data loaded. Please check the file: {task.dataset_path}")
            return

        print(f"Starting evaluation for {task_name}, Total tasks: {len(problems)} (Workers: {self.config.max_workers})...")
        print(f"Model: {self.config.model_name}")
        
        start_time_wall = time.time()
        passed_count = 0
        total_duration = 0
        total_tokens = 0
        
        with Logger(self.config.model_name, task_name) as logger:
            logger.write_header(task.log_columns)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_item = {
                    executor.submit(task.process_item, item, self.llm_client): item 
                    for item in problems
                }
                
                with tqdm(total=len(problems), desc="Progress") as pbar:
                    for future in concurrent.futures.as_completed(future_to_item):
                        try:
                            result = future.result()
                            logger.log_result(result, task.log_columns)
                            
                            if result.get("status") == "PASSED":
                                passed_count += 1
                            
                            total_duration += result.get("duration_raw", 0)
                            total_tokens += result.get("tokens", 0)
                            
                        except Exception as e:
                            print(f"Task execution exception: {e}")
                        finally:
                            pbar.update(1)
            
            end_time_wall = time.time()
            wall_time = end_time_wall - start_time_wall
            
            self._print_summary(logger, passed_count, len(problems), total_duration, total_tokens, wall_time)

    def _print_summary(self, logger, passed, total, total_duration, total_tokens, wall_time):
        avg_duration = total_duration / total if total > 0 else 0
        avg_tokens = total_tokens / total if total > 0 else 0
        
        summary = (
            f"\n{'=' * 50}\n"
            f"Evaluation Completed: {passed}/{total} tasks passed (Pass@{self.config.pass_k})\n"
            f"Pass@{self.config.pass_k}: {passed/total:.2%}\n"
            f"Total Task Duration: {format_time(total_duration)} (Avg: {avg_duration:.2f}s/task)\n"
            f"Wall Clock Time: {format_time(wall_time)}\n"
            f"Total Tokens: {total_tokens} (Avg: {avg_tokens:.1f}/task)\n"
            f"Detailed results saved to: {logger.get_log_path()}\n"
            f"{'=' * 50}\n"
        )
        logger.log_summary(summary)
