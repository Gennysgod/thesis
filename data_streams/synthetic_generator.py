# data_streams/synthetic_generator.py

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from abc import ABC, abstractmethod
from skmultiflow.data import SineGenerator as SKSineGenerator
from skmultiflow.data import SEAGenerator as SKSEAGenerator
from skmultiflow.data import HyperplaneGenerator as SKHyperplaneGenerator
from skmultiflow.data import RandomRBFGenerator as SKRandomRBFGenerator

class BaseDataGenerator(ABC):
    """Base class for data stream generators with enhanced drift and imbalance control"""
    
    def __init__(self, n_samples: int = 2000, n_features: int = 2,
                 drift_start: int = 1000, drift_end: int = 1500,
                 drift_severity: float = 0.15, drift_type: str = 'gradual',
                 imbalance_ratio: List[float] = [5, 5], random_state: int = 42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.drift_start = drift_start
        self.drift_end = drift_end
        self.drift_severity = drift_severity
        self.drift_type = drift_type
        self.imbalance_ratio = imbalance_ratio
        self.random_state = random_state
        
        np.random.seed(random_state)
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate generator parameters"""
        if self.drift_start >= self.drift_end:
            raise ValueError("drift_start must be less than drift_end")
        if self.drift_end >= self.n_samples:
            raise ValueError("drift_end must be less than n_samples")
        if not 0 <= self.drift_severity <= 1:
            raise ValueError("drift_severity must be between 0 and 1")
        if sum(self.imbalance_ratio) == 0:
            raise ValueError("imbalance_ratio cannot sum to zero")

    @abstractmethod
    def _generate_base_stream(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base data stream using library implementation"""
        pass

    @abstractmethod
    def _apply_concept_drift(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply concept drift to the data stream"""
        pass

    def _apply_class_imbalance_by_generation(self) -> pd.DataFrame:
        """Generate data with target class imbalance by controlling generation process"""
        target_ratio = np.array(self.imbalance_ratio) / sum(self.imbalance_ratio)
        target_class_1_ratio = target_ratio[1]
        
        # Generate more data than needed, then select to achieve exact count and ratio
        oversample_factor = 3  # Generate 3x more data
        temp_n_samples = self.n_samples * oversample_factor
        
        # Temporarily increase n_samples for generation
        original_n_samples = self.n_samples
        self.n_samples = temp_n_samples
        
        # Generate base stream
        X, y = self._generate_base_stream()
        
        # Apply concept drift
        y_drifted = self._apply_concept_drift(X, y)
        
        # Restore original n_samples
        self.n_samples = original_n_samples
        
        # Now select exactly n_samples with target ratio
        final_indices = self._select_balanced_indices(y_drifted, target_class_1_ratio)
        
        X_final = X[final_indices]
        y_final = y_drifted[final_indices]
        
        # Create time index
        time_index = np.arange(len(X_final))
        
        # Create DataFrame
        columns = [f'x{i+1}' for i in range(X_final.shape[1])] + ['y', 'time_index']
        data = np.column_stack([X_final, y_final, time_index])
        
        return pd.DataFrame(data, columns=columns)

    def _select_balanced_indices(self, y: np.ndarray, target_class_1_ratio: float) -> np.ndarray:
        """Select indices to achieve target sample count and class ratio"""
        # Calculate target counts
        target_class_1_count = int(self.n_samples * target_class_1_ratio)
        target_class_0_count = self.n_samples - target_class_1_count
        
        # Ensure minimum counts
        target_class_1_count = max(1, target_class_1_count)
        target_class_0_count = max(1, target_class_0_count)
        
        # Get indices for each class
        class_0_indices = np.where(y == 0)[0]
        class_1_indices = np.where(y == 1)[0]
        
        # Sample from each class
        selected_indices = []
        
        if len(class_0_indices) >= target_class_0_count:
            selected_0 = np.random.choice(class_0_indices, target_class_0_count, replace=False)
            selected_indices.extend(selected_0)
        else:
            # If not enough class 0, use all and sample with replacement
            selected_0 = np.random.choice(class_0_indices, target_class_0_count, replace=True)
            selected_indices.extend(selected_0)
        
        if len(class_1_indices) >= target_class_1_count:
            selected_1 = np.random.choice(class_1_indices, target_class_1_count, replace=False)
            selected_indices.extend(selected_1)
        else:
            # If not enough class 1, use all and sample with replacement
            selected_1 = np.random.choice(class_1_indices, target_class_1_count, replace=True)
            selected_indices.extend(selected_1)
        
        # Sort to maintain some temporal order
        selected_indices = sorted(selected_indices)
        
        return np.array(selected_indices)

    def generate_stream(self) -> pd.DataFrame:
        """Generate complete data stream with concept drift and class imbalance"""
        return self._apply_class_imbalance_by_generation()

class SINE1Generator(BaseDataGenerator):
    """SINE1 generator with proper concept drift implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(n_features=2, **kwargs)

    def _generate_base_stream(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base SINE1 stream"""
        generator = SKSineGenerator()
        X, y = generator.next_sample(self.n_samples)
        return np.array(X), np.array(y).flatten()

    def _apply_concept_drift(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply concept drift by shifting sine function"""
        y_drifted = y.copy()
        
        if self.drift_start < self.drift_end:
            shift = self.drift_severity * np.pi
            
            if self.drift_type == 'sudden':
                change_indices = np.arange(self.drift_start, len(X))
                y_drifted[change_indices] = (
                    X[change_indices, 1] > np.sin(X[change_indices, 0] + shift)
                ).astype(int)
                
            elif self.drift_type == 'gradual':
                for i in range(self.drift_start, min(self.drift_end, len(X))):
                    progress = (i - self.drift_start) / (self.drift_end - self.drift_start)
                    current_shift = progress * shift
                    y_drifted[i] = int(X[i, 1] > np.sin(X[i, 0] + current_shift))
                
                after_indices = np.arange(self.drift_end, len(X))
                y_drifted[after_indices] = (
                    X[after_indices, 1] > np.sin(X[after_indices, 0] + shift)
                ).astype(int)
        
        return y_drifted

class SEAGenerator(BaseDataGenerator):
    """SEA generator with proper concept drift implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(n_features=3, **kwargs)

    def _generate_base_stream(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base SEA stream"""
        generator = SKSEAGenerator(classification_function=1)
        X, y = generator.next_sample(self.n_samples)
        return np.array(X), np.array(y).flatten()

    def _apply_concept_drift(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply concept drift by changing threshold"""
        y_drifted = y.copy()
        
        original_threshold = 8.0
        threshold_change = self.drift_severity * 4.0
        new_threshold = original_threshold + threshold_change
        
        if self.drift_start < self.drift_end:
            if self.drift_type == 'sudden':
                change_indices = np.arange(self.drift_start, len(X))
                y_drifted[change_indices] = (
                    X[change_indices, 0] + X[change_indices, 1] <= new_threshold
                ).astype(int)
                
            elif self.drift_type == 'gradual':
                for i in range(self.drift_start, min(self.drift_end, len(X))):
                    progress = (i - self.drift_start) / (self.drift_end - self.drift_start)
                    current_threshold = original_threshold + progress * threshold_change
                    y_drifted[i] = int(X[i, 0] + X[i, 1] <= current_threshold)
                
                after_indices = np.arange(self.drift_end, len(X))
                y_drifted[after_indices] = (
                    X[after_indices, 0] + X[after_indices, 1] <= new_threshold
                ).astype(int)
        
        return y_drifted

# 其他生成器保持相同的修改模式...
class CircleGenerator(BaseDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(n_features=2, **kwargs)
        self.radius_before = 0.5
        self.radius_after = 0.5 + self.drift_severity * 0.3

    def _generate_base_stream(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.random.uniform(0, 1, size=(self.n_samples, 2))
        center = np.array([0.5, 0.5])
        distances = np.linalg.norm(X - center, axis=1)
        y = (distances < self.radius_before).astype(int)
        return X, y

    def _apply_concept_drift(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_drifted = y.copy()
        center = np.array([0.5, 0.5])
        
        if self.drift_start < self.drift_end:
            change_indices = np.arange(self.drift_start, len(X))
            distances = np.linalg.norm(X[change_indices] - center, axis=1)
            y_drifted[change_indices] = (distances < self.radius_after).astype(int)
        
        return y_drifted

class HyperplaneGenerator(BaseDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(n_features=4, **kwargs)

    def _generate_base_stream(self) -> Tuple[np.ndarray, np.ndarray]:
        generator = SKHyperplaneGenerator(n_features=self.n_features)
        X, y = generator.next_sample(self.n_samples)
        return np.array(X), np.array(y).flatten()

    def _apply_concept_drift(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_drifted = y.copy()
        weights_before = np.array([0.3, 0.3, 0.2, 0.2])
        weights_after = np.array([0.2, 0.2, 0.3, 0.3])
        
        if self.drift_start < self.drift_end:
            change_indices = np.arange(self.drift_start, len(X))
            y_drifted[change_indices] = (
                np.dot(X[change_indices], weights_after) > 0
            ).astype(int)
        
        return y_drifted

class RandomRBFGenerator(BaseDataGenerator):
    def __init__(self, n_centers: int = 5, **kwargs):
        super().__init__(n_features=3, **kwargs)
        self.n_centers = n_centers

    def _generate_base_stream(self) -> Tuple[np.ndarray, np.ndarray]:
        generator = SKRandomRBFGenerator(
            n_features=self.n_features,
            n_centroids=self.n_centers
        )
        X, y = generator.next_sample(self.n_samples)
        return np.array(X), np.array(y).flatten()

    def _apply_concept_drift(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_drifted = y.copy()
        
        if self.drift_start < self.drift_end:
            drift_indices = np.arange(self.drift_start, len(y))
            flip_probability = self.drift_severity
            
            for i in drift_indices:
                if np.random.random() < flip_probability:
                    y_drifted[i] = 1 - y_drifted[i]
        
        return y_drifted

class DataStreamFactory:
    """Factory for creating data stream generators"""
    
    generators = {
        'SINE1': SINE1Generator,
        'SEA': SEAGenerator,
        'Circle': CircleGenerator,
        'Sine': SINE1Generator,
        'Hyperplane': HyperplaneGenerator,
        'RandomRBF': RandomRBFGenerator
    }

    @classmethod
    def create_generator(cls, generator_type: str, **kwargs) -> BaseDataGenerator:
        if generator_type not in cls.generators:
            raise ValueError(f"Unknown generator type: {generator_type}")
        return cls.generators[generator_type](**kwargs)

    @classmethod
    def generate_pretraining_data(cls, n_streams_per_type: int = 1000,
                                drift_types: List[str] = ['sudden', 'gradual'],
                                drift_severity_range: Tuple[float, float] = (0.01, 0.3),
                                random_state: int = 42) -> List[Tuple[np.ndarray, Dict]]:
        np.random.seed(random_state)
        training_data = []
        
        generator_types = ['Circle', 'SINE1']
        stream_id = 0
        streams_per_combination = n_streams_per_type // (len(generator_types) * len(drift_types))
        
        for generator_type in generator_types:
            for drift_type in drift_types:
                for i in range(streams_per_combination):
                    drift_severity = np.random.uniform(*drift_severity_range)
                    drift_start = np.random.randint(200, 400)
                    drift_duration = np.random.randint(100, 200)
                    drift_end = min(drift_start + drift_duration, 800)
                    
                    try:
                        generator = cls.create_generator(
                            generator_type,
                            n_samples=1000,
                            drift_start=drift_start,
                            drift_end=drift_end,
                            drift_severity=drift_severity,
                            drift_type=drift_type,
                            imbalance_ratio=[5, 5],
                            random_state=random_state + stream_id
                        )
                        
                        stream_df = generator.generate_stream()
                        feature_cols = [col for col in stream_df.columns if col.startswith('x')]
                        X = stream_df[feature_cols].values
                        y = stream_df['y'].values
                        
                        from models.classifier import generate_accuracy_sequence
                        accuracy_sequence = generate_accuracy_sequence(X, y)
                        
                        Ds = drift_start / 1000.0
                        De = drift_end / 1000.0
                        Dv = drift_severity
                        Dt = 1 if drift_type == 'gradual' else 0
                        
                        quadruple = {
                            'Ds': Ds, 'De': De, 'Dv': Dv, 'Dt': Dt,
                            'generator_type': generator_type,
                            'drift_type': drift_type
                        }
                        
                        training_data.append((accuracy_sequence, quadruple))
                        stream_id += 1
                        
                    except Exception as e:
                        print(f"Warning: Failed to generate stream {stream_id}: {e}")
                        continue
        
        return training_data