import os
import time
import tempfile
import subprocess
import sys
from typing import Tuple

def load_dotenv(dotenv_path: str = ".env", override: bool = False) -> bool:
    if not os.path.exists(dotenv_path):
        return False

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue

            key, val = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            val = val.strip()
            if val and len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
                val = val[1:-1]

            if not override and key in os.environ:
                continue
            os.environ[key] = val

    return True

def _safe_path_segment(name: str) -> str:
    if name is None:
        return "unknown"
    s = str(name).strip()
    if not s:
        return "unknown"

    invalid = '<>:"/\\|?*'
    cleaned = []
    for ch in s:
        o = ord(ch)
        if o < 32 or ch in invalid:
            cleaned.append("_")
        else:
            cleaned.append(ch)

    s = "".join(cleaned)
    s = s.rstrip(" .")
    return s if s else "unknown"

def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class Logger:
    def __init__(self, model_name: str, task_name: str):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.model_name = model_name
        self.task_name = task_name

        safe_model_name = _safe_path_segment(model_name)
        self.output_dir = os.path.join("model_test", safe_model_name)
        os.makedirs(self.output_dir, exist_ok=True)

        safe_task_name = _safe_path_segment(task_name)
        self.log_path = os.path.join(self.output_dir, f"{safe_task_name}_{self.timestamp}.log")
        self.log_handle = None

    def __enter__(self):
        self.log_handle = open(self.log_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_handle:
            self.log_handle.close()

    def write_header(self, columns: list):
        if self.log_handle:
            self.log_handle.write("# Individual Test Results\n")
            self.log_handle.write("# " + "\t".join(columns) + "\n")
            self.log_handle.flush()

    def log_result(self, data: dict, columns: list):
        if self.log_handle:
            row = [str(data.get(col, "")) for col in columns]
            self.log_handle.write("\t".join(row) + "\n")
            self.log_handle.flush()

    def log_message(self, message: str, level: str = "INFO"):
        if self.log_handle:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"# [{timestamp}] [{level}] {message}\n"
            self.log_handle.write(log_entry)
            self.log_handle.flush()

    def log_summary(self, summary: str):
        if self.log_handle:
            self.log_handle.write("\n" + "=" * 60 + "\n")
            self.log_handle.write(summary + "\n")
            self.log_handle.flush()
        return

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
