
from typing import Dict, List, Any
import os

# Main experiment configuration
EXPERIMENT_CONFIG = {
    # Data groups configuration
    'data_groups': {
        'Group A1': {
            'attributes': 2,
            'generator': 'SINE1', 
            'imbalance_ratio': [5, 5],
            'description': '2D balanced data'
        },
        'Group A2': {
            'attributes': 2,
            'generator': 'SINE1',
            'imbalance_ratio': [7, 3],
            'description': '2D moderate imbalance'
        },
        'Group A3': {
            'attributes': 2,
            'generator': 'SINE1',
            'imbalance_ratio': [9, 1],
            'description': '2D high imbalance'
        },
        'Group B1': {
            'attributes': 3,
            'generator': 'SEA',
            'imbalance_ratio': [5, 5],
            'description': '3D balanced data'
        },
        'Group B2': {
            'attributes': 3,
            'generator': 'SEA',
            'imbalance_ratio': [7, 3],
            'description': '3D moderate imbalance'
        },
        'Group B3': {
            'attributes': 3,
            'generator': 'SEA',
            'imbalance_ratio': [9, 1],
            'description': '3D high imbalance'
        }
    },
    
    # Detector configuration
    'detectors': ['ADWIN', 'DDM', 'QuadCDD'],
    
    # Evaluation metrics
    'metrics': [
        'Drift Detection Delay',
        'False Positive Rate', 
        'Missed Drift Count',
        'Drift Detection Recall'
    ],
    
    # Drift parameters (consistent across all data groups)
    'drift_params': {
        'Ds': 1000,           # Drift start time
        'De': 1500,           # Drift end time  
        'Dv': 0.15,           # Drift severity (medium)
        'Dt': 'gradual'       # Drift type
    },
    
    # Experiment execution parameters
    'n_runs': 20,                           # Number of runs per combination
    'random_seeds': list(range(42, 62)),    # Seeds for reproducibility
    'stream_length': 2000,                  # Total stream length
    
    # Detection parameters
    'detection_params': {
        'window_size': 100,                 # Sliding window size
        'min_detection_window': 200,        # Minimum samples before detection
        'tolerance_window': 50,             # Detection tolerance window
    },
    
    # Quality thresholds
    'quality_thresholds': {
        'fpr_threshold': 0.05,              # Maximum acceptable FPR
        'delay_threshold': 50,              # Maximum acceptable delay
        'min_recall': 0.8                   # Minimum acceptable recall
    }
}

# Pre-training configuration for QuadCDD
PRETRAIN_CONFIG = {
    # Data generation for pre-training
    'generators': ['Circle', 'Sine', 'Hyperplane', 'RandomRBF'],
    'drift_types': ['sudden', 'gradual', 'early-abrupt'],
    'streams_per_type': 1000,
    'total_streams': 4000,
    'stream_length': 1000,
    
    # Drift parameter ranges for training diversity
    'drift_severity_range': [0.01, 0.3],
    'drift_start_range': [200, 600],
    'drift_duration_range': [100, 300],
    'drift_threshold': 0.2,
    
    # Model architecture
    'model_config': {
        'input_size': 1,
        'hidden_size_1': 128,
        'hidden_size_2': 64,
        'output_size': 4,
        'num_layers': 1,
        'dropout': 0.2
    },
    
    # Training parameters
    'training_config': {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 50,
        'early_stopping_patience': 15,
        'weight_decay': 1e-5,
        'train_val_split': 0.8
    },
    
    # Fine-tuning parameters
    'finetuning_config': {
        'learning_rate': 1e-2,
        'epochs': 10,
        'batch_size': 16
    }
}

# Detector-specific configurations
DETECTOR_CONFIGS = {
    'ADWIN': {
        'delta': 0.002,
        'description': 'Adaptive Windowing algorithm for drift detection'
    },
    'DDM': {
        'warning_level': 2.0,
        'drift_level': 3.0,
        'description': 'Drift Detection Method based on error rate monitoring'
    },
    'QuadCDD': {
        'window_size': 100,
        'min_detection_window': 200,
        'detection_threshold': 0.2,
        'model_path': 'models/pretrained_models/quadcdd_pretrained.pth',
        'description': 'Quadruple-based Concept Drift Detection'
    }
}

# File paths and directories
PATHS = {
    'data_dir': 'D:/study/project/data/experiment1/',
    'results_dir': 'results/',
    'raw_results_dir': 'results/raw_results/',
    'aggregated_results_dir': 'results/aggregated_results/',
    'figures_dir': 'results/figures/',
    'models_dir': 'models/pretrained_models/',
    'logs_dir': 'logs/'
}

# Output file naming conventions
OUTPUT_FILES = {
    'raw_results': 'experiment1_raw_results.csv',
    'aggregated_results': 'experiment1_aggregated_results.csv',
    'summary_stats': 'experiment1_summary_statistics.json',
    'quality_report': 'experiment1_quality_report.json',
    'experiment_log': 'experiment1.log'
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'format': 'png',
    'style': 'seaborn-v0_8',
    
    # Color schemes
    'detector_colors': {
        'ADWIN': '#1f77b4',
        'DDM': '#ff7f0e',
        'QuadCDD': '#2ca02c'
    },
    
    'metric_colors': {
        'Drift Detection Delay': '#d62728',
        'False Positive Rate': '#9467bd',
        'Missed Drift Count': '#8c564b',
        'Drift Detection Recall': '#e377c2'
    },
    
    'imbalance_colors': {
        '5:5': '#2ca02c',    # Green for balanced
        '7:3': '#ff7f0e',    # Orange for moderate
        '9:1': '#d62728'     # Red for high imbalance
    }
}

# Experiment execution configuration
EXECUTION_CONFIG = {
    'parallel_processing': False,    # Set to True for parallel execution
    'n_processes': 4,               # Number of parallel processes
    'progress_bar': True,           # Show progress bars
    'save_intermediate': True,      # Save intermediate results
    'verbose': True,                # Verbose logging
    'seed_base': 42                 # Base seed for reproducibility
}

# Statistical analysis configuration
STATISTICS_CONFIG = {
    'confidence_level': 0.95,
    'significance_level': 0.05,
    'statistical_tests': [
        'anova',           # One-way ANOVA for group comparisons
        'tukey_hsd',       # Tukey's HSD for post-hoc analysis
        'kruskal_wallis'   # Non-parametric alternative
    ],
    'effect_size_measures': [
        'eta_squared',     # Effect size for ANOVA
        'cohens_d'         # Effect size for pairwise comparisons
    ]
}


class ExperimentConfigManager:
    """Manager class for experiment configuration"""
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config = EXPERIMENT_CONFIG.copy()
        self.pretrain_config = PRETRAIN_CONFIG.copy()
        self.detector_configs = DETECTOR_CONFIGS.copy()
        self.paths = PATHS.copy()
        self.output_files = OUTPUT_FILES.copy()
        self.viz_config = VISUALIZATION_CONFIG.copy()
        self.exec_config = EXECUTION_CONFIG.copy()
        self.stats_config = STATISTICS_CONFIG.copy()
        
        # Apply any overrides
        if config_override:
            self._apply_overrides(config_override)
    
    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides"""
        for key, value in overrides.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
    
    def get_data_group_config(self, group_name: str) -> Dict[str, Any]:
        """Get configuration for specific data group"""
        if group_name not in self.config['data_groups']:
            raise ValueError(f"Unknown data group: {group_name}")
        return self.config['data_groups'][group_name]
    
    def get_detector_config(self, detector_name: str) -> Dict[str, Any]:
        """Get configuration for specific detector"""
        if detector_name not in self.detector_configs:
            raise ValueError(f"Unknown detector: {detector_name}")
        return self.detector_configs[detector_name]
    
    def get_experiment_combinations(self) -> List[Dict[str, Any]]:
        """Get all experiment combinations to run"""
        combinations = []
        
        for detector in self.config['detectors']:
            for group_name, group_config in self.config['data_groups'].items():
                for run_id in range(self.config['n_runs']):
                    combination = {
                        'detector': detector,
                        'data_group': group_name,
                        'run_id': run_id,
                        'seed': self.config['random_seeds'][run_id],
                        'group_config': group_config,
                        'detector_config': self.get_detector_config(detector)
                    }
                    combinations.append(combination)
        
        return combinations
    
    def get_total_experiments(self) -> int:
        """Get total number of experiments to run"""
        return len(self.config['detectors']) * len(self.config['data_groups']) * self.config['n_runs']
    
    def create_output_directories(self):
        """Create all necessary output directories"""
        from utils.common import ensure_directory
        
        for path in self.paths.values():
            ensure_directory(path)
    
    def save_config(self, filepath: str):
        """Save complete configuration to file"""
        import json
        
        config_data = {
            'experiment_config': self.config,
            'pretrain_config': self.pretrain_config,
            'detector_configs': self.detector_configs,
            'paths': self.paths,
            'output_files': self.output_files,
            'visualization_config': self.viz_config,
            'execution_config': self.exec_config,
            'statistics_config': self.stats_config
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_config(self, filepath: str):
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        self.config = config_data.get('experiment_config', self.config)
        self.pretrain_config = config_data.get('pretrain_config', self.pretrain_config)
        self.detector_configs = config_data.get('detector_configs', self.detector_configs)
        self.paths = config_data.get('paths', self.paths)
        self.output_files = config_data.get('output_files', self.output_files)
        self.viz_config = config_data.get('visualization_config', self.viz_config)
        self.exec_config = config_data.get('execution_config', self.exec_config)
        self.stats_config = config_data.get('statistics_config', self.stats_config)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check if all required paths exist
        required_dirs = ['models_dir']
        for dir_key in required_dirs:
            if dir_key in self.paths:
                if not os.path.exists(self.paths[dir_key]):
                    issues.append(f"Required directory does not exist: {self.paths[dir_key]}")
        
        # Check if QuadCDD model exists
        quadcdd_config = self.detector_configs.get('QuadCDD', {})
        model_path = quadcdd_config.get('model_path')
        if model_path and not os.path.exists(model_path):
            issues.append(f"QuadCDD pre-trained model not found: {model_path}")
        
        # Check if number of seeds matches number of runs
        if len(self.config['random_seeds']) < self.config['n_runs']:
            issues.append(f"Not enough random seeds ({len(self.config['random_seeds'])}) for number of runs ({self.config['n_runs']})")
        
        # Check detector configurations
        for detector in self.config['detectors']:
            if detector not in self.detector_configs:
                issues.append(f"Configuration missing for detector: {detector}")
        
        return issues
    
    def print_summary(self):
        """Print configuration summary"""
        print("="*60)
        print("EXPERIMENT CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"Data Groups: {len(self.config['data_groups'])}")
        for name, config in self.config['data_groups'].items():
            ratio_str = f"{config['imbalance_ratio'][0]}:{config['imbalance_ratio'][1]}"
            print(f"  {name}: {config['generator']} ({config['attributes']}D, {ratio_str})")
        
        print(f"\nDetectors: {', '.join(self.config['detectors'])}")
        print(f"Runs per combination: {self.config['n_runs']}")
        print(f"Total experiments: {self.get_total_experiments()}")
        
        print(f"\nDrift Parameters:")
        for key, value in self.config['drift_params'].items():
            print(f"  {key}: {value}")
        
        print(f"\nQuality Thresholds:")
        for key, value in self.config['quality_thresholds'].items():
            print(f"  {key}: {value}")
        
        print(f"\nOutput Directory: {self.paths['results_dir']}")
        print("="*60)


# Global configuration instance
config_manager = ExperimentConfigManager()


def get_config_manager(config_override: Dict[str, Any] = None) -> ExperimentConfigManager:
    """Get configuration manager instance"""
    if config_override:
        return ExperimentConfigManager(config_override)
    return config_manager


def validate_experiment_setup() -> bool:
    """Validate complete experiment setup"""
    issues = config_manager.validate_config()
    
    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Configuration validation passed ✓")
        return True


def create_sample_config_file():
    """Create sample configuration file for customization"""
    sample_config = {
        'n_runs': 5,  # Reduced for testing
        'quality_thresholds': {
            'fpr_threshold': 0.1,  # More lenient for testing
            'delay_threshold': 100
        }
    }
    
    config_manager.save_config('sample_experiment_config.json')
    print("Sample configuration saved to 'sample_experiment_config.json'")


if __name__ == "__main__":
    # Print configuration summary
    config_manager.print_summary()
    
    # Validate setup
    validate_experiment_setup()
    
    # Create sample config file
    create_sample_config_file()