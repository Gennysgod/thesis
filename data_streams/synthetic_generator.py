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
        if not 0 <= self.drift_severity <= 2.0:  # Extended range for higher severity
            raise ValueError("drift_severity must be between 0 and 2.0")
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

    def _apply_class_imbalance_by_concept_aware_generation(self) -> pd.DataFrame:
        """Generate data with target class imbalance while preserving concept relationships"""
        target_ratio = np.array(self.imbalance_ratio) / sum(self.imbalance_ratio)
        target_class_1_ratio = target_ratio[1]
        
        # Generate data in segments to maintain temporal order
        pre_drift_samples = self.drift_start
        drift_samples = self.drift_end - self.drift_start
        post_drift_samples = self.n_samples - self.drift_end
        
        all_data = []
        time_index = 0
        
        # Generate each segment separately to maintain concept relationships
        for segment_samples, segment_start, segment_end in [
            (pre_drift_samples, 0, self.drift_start),
            (drift_samples, self.drift_start, self.drift_end), 
            (post_drift_samples, self.drift_end, self.n_samples)
        ]:
            if segment_samples <= 0:
                continue
                
            # Generate more data for this segment
            oversample_factor = 2
            temp_samples = max(segment_samples * oversample_factor, 100)
            
            # Temporarily adjust parameters for segment generation
            original_n_samples = self.n_samples
            original_drift_start = self.drift_start
            original_drift_end = self.drift_end
            
            self.n_samples = temp_samples
            self.drift_start = 0 if segment_start == 0 else int(temp_samples * 0.3)
            self.drift_end = temp_samples if segment_end == original_n_samples else int(temp_samples * 0.7)
            
            # Generate segment data
            X, y = self._generate_base_stream()
            
            # Apply drift if this is the appropriate segment
            if segment_start >= original_drift_start:
                y_drifted = self._apply_concept_drift(X, y)
            else:
                y_drifted = y.copy()
            
            # Restore original parameters
            self.n_samples = original_n_samples
            self.drift_start = original_drift_start  
            self.drift_end = original_drift_end
            
            # Select samples to achieve target ratio for this segment
            selected_indices = self._select_balanced_indices_preserve_order(
                y_drifted, target_class_1_ratio, segment_samples
            )
            
            X_segment = X[selected_indices]
            y_segment = y_drifted[selected_indices]
            
            # Add to complete dataset with correct time indices
            for i, (x_row, y_val) in enumerate(zip(X_segment, y_segment)):
                data_point = list(x_row) + [y_val, time_index]
                all_data.append(data_point)
                time_index += 1
        
        # Create DataFrame
        columns = [f'x{i+1}' for i in range(X.shape[1])] + ['y', 'time_index']
        return pd.DataFrame(all_data, columns=columns)
    
    def _select_balanced_indices_preserve_order(self, y: np.ndarray, 
                                              target_class_1_ratio: float, 
                                              target_count: int) -> np.ndarray:
        """Select indices to achieve target ratio while preserving temporal order"""
        target_class_1_count = int(target_count * target_class_1_ratio)
        target_class_0_count = target_count - target_class_1_count
        
        # Ensure minimum counts
        target_class_1_count = max(1, target_class_1_count)
        target_class_0_count = max(1, target_class_0_count)
        
        # Get indices for each class in order
        class_0_indices = np.where(y == 0)[0]
        class_1_indices = np.where(y == 1)[0]
        
        # Sample evenly across the time series to maintain temporal distribution
        selected_indices = []
        
        if len(class_0_indices) > 0:
            if len(class_0_indices) >= target_class_0_count:
                # Sample evenly across available indices
                step = len(class_0_indices) / target_class_0_count
                selected_0 = [class_0_indices[int(i * step)] for i in range(target_class_0_count)]
            else:
                # Use all available and repeat if necessary
                repeats = target_class_0_count // len(class_0_indices) + 1
                selected_0 = np.tile(class_0_indices, repeats)[:target_class_0_count]
            selected_indices.extend(selected_0)
        
        if len(class_1_indices) > 0:
            if len(class_1_indices) >= target_class_1_count:
                step = len(class_1_indices) / target_class_1_count  
                selected_1 = [class_1_indices[int(i * step)] for i in range(target_class_1_count)]
            else:
                repeats = target_class_1_count // len(class_1_indices) + 1
                selected_1 = np.tile(class_1_indices, repeats)[:target_class_1_count]
            selected_indices.extend(selected_1)
        
        # Sort to maintain temporal order
        return np.array(sorted(selected_indices))

    def generate_stream(self) -> pd.DataFrame:
        """Generate complete data stream with concept drift and class imbalance"""
        return self._apply_class_imbalance_by_concept_aware_generation()

class SINE1Generator(BaseDataGenerator):
    """SINE1 generator with enhanced concept drift implementation"""
    
    def __init__(self, **kwargs):
        # Set default higher drift severity for SINE1 (0.5)
        if 'drift_severity' not in kwargs:
            kwargs['drift_severity'] = 0.5
        super().__init__(n_features=2, **kwargs)

    def _generate_base_stream(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base SINE1 stream"""
        generator = SKSineGenerator()
        X, y = generator.next_sample(self.n_samples)
        return np.array(X), np.array(y).flatten()

    def _apply_concept_drift(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply concept drift by shifting sine function with higher severity"""
        y_drifted = y.copy()
        
        if self.drift_start < self.drift_end:
            # Use higher severity (0.5 * π instead of 0.15 * π)
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
    """SEA generator with enhanced concept drift implementation"""
    
    def __init__(self, **kwargs):
        # Set default higher drift severity for SEA (0.35 -> threshold_change = 3.5)
        if 'drift_severity' not in kwargs:
            kwargs['drift_severity'] = 0.35
        super().__init__(n_features=3, **kwargs)

    def _generate_base_stream(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base SEA stream"""
        generator = SKSEAGenerator(classification_function=1)
        X, y = generator.next_sample(self.n_samples)
        return np.array(X), np.array(y).flatten()

    def _apply_concept_drift(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply concept drift by changing threshold with higher severity"""
        y_drifted = y.copy()
        
        original_threshold = 8.0
        # Use threshold_change = 3.5 (drift_severity * 10)
        threshold_change = self.drift_severity * 10.0
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

class CircleGenerator(BaseDataGenerator):
    def __init__(self, **kwargs):
        # Set default higher drift severity for Circle (1.0)
        if 'drift_severity' not in kwargs:
            kwargs['drift_severity'] = 1.0
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