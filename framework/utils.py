import os
import time
import tempfile
import subprocess
import sys
from typing import Tuple

def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class Logger:
    def __init__(self, model_name: str, task_name: str):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Create a session directory for this run
        self.session_dir = os.path.join("model_test", model_name, f"{task_name}_{self.timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 1. Result File (TSV for easy parsing) - contains only metrics
        self.result_path = os.path.join(self.session_dir, "results.tsv")
        self.result_handle = None
        
        # 2. Log File (Text) - contains process details, errors, summaries
        self.log_path = os.path.join(self.session_dir, "execution.log")
        self.log_handle = None

    def __enter__(self):
        self.result_handle = open(self.result_path, "w", encoding="utf-8")
        self.log_handle = open(self.log_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.result_handle:
            self.result_handle.close()
        if self.log_handle:
            self.log_handle.close()

    def write_header(self, columns: list):
        if self.result_handle:
            self.result_handle.write("\t".join(columns) + "\n")
            self.result_handle.flush()

    def log_result(self, data: dict, columns: list):
        if self.result_handle:
            # Only write to result file
            row = [str(data.get(col, "")) for col in columns]
            self.result_handle.write("\t".join(row) + "\n")
            self.result_handle.flush()
    
    def log_message(self, message: str, level: str = "INFO"):
        """Write detailed logs to execution.log"""
        if self.log_handle:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}\n"
            self.log_handle.write(log_entry)
            self.log_handle.flush()
    
    def log_summary(self, summary: str):
        # Summary goes to log file and console
        if self.log_handle:
            self.log_handle.write("\n" + summary + "\n")
            self.log_handle.flush()
        print(summary)

    def get_result_path(self):
        return os.path.abspath(self.result_path)

    def get_log_path(self):
        return os.path.abspath(self.log_path)

class CodeExecutor:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, code: str) -> Tuple[str, str]:
        """
        Execute Python code and return (status, output/error)
        """
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
                tmp_file_path = tmp_file.name
                tmp_file.write(code)
            
            # Execute
            result = subprocess.run(
                [sys.executable, tmp_file_path],
                capture_output=True,
                timeout=self.timeout
            )
            
            stderr = result.stderr.decode('utf-8', errors='ignore')
            stdout = result.stdout.decode('utf-8', errors='ignore')
            
            if result.returncode == 0:
                return "PASSED", ""
            else:
                error_msg = stderr.strip() if stderr else (stdout.strip() if stdout else "Unknown Error")
                return "FAILED", error_msg.replace("\n", " | ")
                
        except subprocess.TimeoutExpired:
            return "TIMEOUT", "Execution Timed Out"
        except Exception as e:
            return "EXECUTION_ERROR", str(e)
        finally:
            # Cleanup
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except:
                    pass
