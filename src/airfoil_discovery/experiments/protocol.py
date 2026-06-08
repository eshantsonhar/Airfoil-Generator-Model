import json
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ExperimentConfig:
    reynolds: float
    lift_target: float
    seed: int
    fidelity_levels: int = 3
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

class ExperimentReporter:
    """
    Auto-generates report data for publication.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        
    def log_iteration(self, iter_data: Dict[str, Any]):
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(iter_data) + "\n")
