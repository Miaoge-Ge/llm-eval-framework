import os
import yaml
from typing import Dict, Any, Optional

class ConfigManager:
    _instance = None
    _config: Dict[str, Any] = {}
    
    SETTINGS_FILE = "settings.yaml"
    REGISTRY_FILE = "registry.yaml"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            print(f"Warning: Configuration file not found: {path}")
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing {path}: {e}")

    def _load_config(self):
        self._settings = self._load_yaml(self.SETTINGS_FILE)
        self._registry = self._load_yaml(self.REGISTRY_FILE)
        
        self._config = {**self._registry, **self._settings}

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get_selected_model_config(self) -> Dict[str, Any]:
        selected_model_key = self._settings.get("selected_model")
        if not selected_model_key:
            raise ValueError(f"No 'selected_model' defined in {self.SETTINGS_FILE}")
        
        # 1. Base Model Info (from Registry)
        models = self._registry.get("models", {})
        model_config = models.get(selected_model_key)
        
        if not model_config:
            raise ValueError(f"Model '{selected_model_key}' not found in 'models' section of {self.REGISTRY_FILE}")
            
        provider_key = model_config.get("provider")
        if not provider_key:
            raise ValueError(f"No 'provider' specified for model '{selected_model_key}'")
            
        providers = self._registry.get("providers", {})
        provider_config = providers.get(provider_key)
        
        if not provider_config:
            raise ValueError(f"Provider '{provider_key}' not found in 'providers' section of {self.REGISTRY_FILE}")
            
        # 2. Merge: Provider -> Model (Registry) -> Settings (Override)
        final_config = provider_config.copy()
        final_config.update(model_config)
        
        # Override with specific runtime settings if present in settings.yaml
        for override_key in ["temperature", "rpm_limit", "tpm_limit"]:
            if override_key in self._settings:
                final_config[override_key] = self._settings[override_key]
        
        return final_config

    def get_dataset_path(self, task_name: str) -> str:
        datasets = self._registry.get("datasets", {})
        path = datasets.get(task_name)
        if not path:
             return ""
        return path

    def get_global_setting(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, self._registry.get(key, default))

    def reload(self):
        self._load_config()
