import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self._setup_paths()

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        logger.info(f"Loaded config from {yaml_path}")
        return cls(config_dict)

    def _setup_paths(self):
        for path in [self.log_dir, self.checkpoint_dir]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> Dict:
        return self.config.get('model', {})

    @property
    def training(self) -> Dict:
        return self.config.get('training', {})

    @property
    def data(self) -> Dict:
        return self.config.get('data', {})

    @property
    def log_dir(self) -> Path:
        return Path(self.config.get('logging', {}).get('log_dir', './results/logs'))

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.config.get('logging', {}).get('checkpoint_dir', './results/checkpoints'))
