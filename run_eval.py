import multiprocessing
import os
import traceback
from framework import EvalConfig, Runner, TaskRegistry, ConfigManager

def main():
    try:
        config_manager = ConfigManager()
        
        task_name = config_manager.get_global_setting("task")
        if not task_name:
            print("Error: No 'task' defined in settings.yaml")
            return

        dataset_path = config_manager.get_dataset_path(task_name)
        if not dataset_path:
            print(f"Error: Dataset path for task '{task_name}' not defined in registry.yaml")
            return
            
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file not found: {dataset_path}")
            return

        print("Initializing configuration...")
        config = EvalConfig()
        
        print(f"Configuration Loaded:")
        print(f"  Task: {task_name}")
        print(f"  Model Profile: {config_manager.get_global_setting('selected_model')}")
        print(f"  Model Name: {config.model_name}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Workers: {config.max_workers}")

        task_cls = TaskRegistry.get(task_name)
        if not task_cls:
            print(f"Error: Unknown task '{task_name}' registered in system.")
            print(f"Available tasks: {TaskRegistry.list_tasks()}")
            return
            
        print("Initializing task...")
        task = task_cls(dataset_path)
        print("Initializing runner...")
        runner = Runner(config)
        print("Starting execution...")
        runner.run(task, task_name)
        print("Execution finished.")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
