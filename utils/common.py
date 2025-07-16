import os
import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, List, Any, Tuple
import logging

def setup_logging(log_file: str = "experiment.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directory(path: str):
    """Ensure directory exists, create if not"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(results: Dict, filename: str, results_dir: str = "results"):
    """Save results to CSV file"""
    ensure_directory(results_dir)
    filepath = os.path.join(results_dir, filename)
    if isinstance(results, dict):
        # Convert dict to DataFrame
        df = pd.DataFrame([results])
    else:
        df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

def load_results(filename: str, results_dir: str = "results") -> pd.DataFrame:
    """Load results from CSV file"""
    filepath = os.path.join(results_dir, filename)
    return pd.read_csv(filepath)

def save_model(model: Any, filename: str, model_dir: str = "models/pretrained_models"):
    """Save model using pickle"""
    ensure_directory(model_dir)
    filepath = os.path.join(model_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filename: str, model_dir: str = "models/pretrained_models") -> Any:
    """Load model using pickle"""
    filepath = os.path.join(model_dir, filename)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def calculate_imbalance_ratio(y: np.ndarray) -> Tuple[float, float]:
    """Calculate class imbalance ratio"""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    if len(unique) == 2:
        ratio_0 = counts[0] / total
        ratio_1 = counts[1] / total
        return ratio_0, ratio_1
    else:
        return 0.5, 0.5

def normalize_time_series(series: np.ndarray) -> np.ndarray:
    """Normalize time series to [0, 1] range"""
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val == min_val:
        return np.zeros_like(series)
    return (series - min_val) / (max_val - min_val)

def create_sliding_window(data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    """Create sliding window from time series data"""
    n = len(data)
    windows = []
    for i in range(0, n - window_size + 1, step):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows)

def interpolate_drift_transition(start_val: float, end_val: float,
                               start_time: int, end_time: int,
                               drift_type: str = 'gradual') -> np.ndarray:
    """Create drift transition values"""
    duration = end_time - start_time
    if duration <= 0:
        return np.array([end_val])
    
    if drift_type == 'sudden':
        # Sudden change at start_time
        return np.full(duration, end_val)
    elif drift_type == 'gradual':
        # Linear interpolation
        return np.linspace(start_val, end_val, duration)
    elif drift_type == 'early-abrupt':
        # Change happens in first 20% of duration
        change_point = max(1, int(0.2 * duration))
        transition = np.full(duration, start_val)
        transition[change_point:] = end_val
        return transition
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")

def generate_class_labels(n_samples: int, imbalance_ratio: List[float],
                        random_state: int = None) -> np.ndarray:
    """Generate class labels with specified imbalance ratio"""
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    if len(imbalance_ratio) != 2:
        raise ValueError("imbalance_ratio must have exactly 2 elements")
    
    if sum(imbalance_ratio) == 0:
        raise ValueError("imbalance_ratio cannot sum to zero")

    # Calculate number of samples for each class
    total_ratio = sum(imbalance_ratio)
    n_positive = int(n_samples * imbalance_ratio[1] / total_ratio)
    n_negative = n_samples - n_positive

    # Ensure we have at least one sample of each class
    n_positive = max(1, min(n_positive, n_samples - 1))
    n_negative = n_samples - n_positive

    # Create labels
    labels = np.concatenate([
        np.zeros(n_negative),
        np.ones(n_positive)
    ])

    # Shuffle labels
    np.random.shuffle(labels)
    return labels.astype(int)

class ExperimentConfig:
    """Configuration manager for experiments"""

    def __init__(self, config_file: str = None):
        self.config = self._load_default_config()
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)

    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'data_groups': {
                'Group A1': {'attributes': 2, 'generator': 'SINE1', 'imbalance_ratio': [5, 5]},
                'Group A2': {'attributes': 2, 'generator': 'SINE1', 'imbalance_ratio': [7, 3]},
                'Group A3': {'attributes': 2, 'generator': 'SINE1', 'imbalance_ratio': [9, 1]},
                'Group B1': {'attributes': 3, 'generator': 'SEA', 'imbalance_ratio': [5, 5]},
                'Group B2': {'attributes': 3, 'generator': 'SEA', 'imbalance_ratio': [7, 3]},
                'Group B3': {'attributes': 3, 'generator': 'SEA', 'imbalance_ratio': [9, 1]}
            },
            'detectors': ['ADWIN', 'DDM', 'QuadCDD'],
            'metrics': ['Drift Detection Delay', 'False Positive Rate',
                       'Missed Drift Count', 'Drift Detection Recall'],
            'drift_params': {
                'Ds': 1000, 'De': 1500,
                'Dv': 0.15,
                'Dt': 'gradual'
            },
            'n_runs': 20,
            'random_seeds': list(range(42, 62)),
            'window_size': 100,
            'min_detection_window': 200,
            'thresholds': {
                'fpr_threshold': 0.05,
                'delay_threshold': 50
            }
        }

    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def save(self, filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)

# Global configuration instance
config = ExperimentConfig()