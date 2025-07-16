import numpy as np
from .base_detector import BaseDetector
from collections import deque
from river import tree

# Try to import skmultiflow DDM
try:
    from skmultiflow.drift_detection import DDM
    SKMULTIFLOW_AVAILABLE = True
    print("✓ skmultiflow DDM imported successfully")
except ImportError:
    SKMULTIFLOW_AVAILABLE = False
    print("✗ skmultiflow DDM not available, using simple fallback")

class DDMDetector(BaseDetector):
    """DDM drift detector with river-based incremental learning"""
    
    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0, **kwargs):
        super().__init__("DDM")
        self.warning_level = warning_level
        self.drift_level = drift_level
        
        # Initialize DDM detector
        if SKMULTIFLOW_AVAILABLE:
            # skmultiflow 0.5.3 DDM uses 'out_control_level' instead of 'drift_level'
            try:
                self.detector = DDM(warning_level=warning_level, out_control_level=drift_level)
                self.backend = 'skmultiflow'
                print(f"✓ DDM initialized with skmultiflow backend (warn={warning_level}, out_control={drift_level})")
            except Exception as e:
                print(f"✗ skmultiflow DDM initialization failed: {e}")
                self._init_simple_ddm()
        else:
            self._init_simple_ddm()
        
        # River-based incremental classifier for prediction
        self.incremental_classifier = tree.HoeffdingTreeClassifier()
        self.samples_seen = 0
        self.min_samples_for_prediction = 10
        
    def _init_simple_ddm(self):
        """Initialize simple DDM implementation as fallback"""
        self.detector = None
        self.backend = 'simple'
        
        # Simple DDM state variables
        self.error_count = 0
        self.sample_count = 0
        self.min_instances = 30
        
        self.p_min = float('inf')
        self.s_min = float('inf')
        
        self.p = 0.0
        self.s = 0.0
        
        self.in_warning_zone = False
        print("✓ DDM initialized with simple fallback backend")
        
    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Update DDM with new data point using river-based prediction"""
        
        # Convert inputs to proper format
        if X.ndim > 1:
            X = X.flatten()
        
        # Convert to dict format for river
        x_dict = {f'x{i}': float(X[i]) for i in range(len(X))}
        
        # Ensure y is scalar int
        if np.isarray(y):
            y_val = int(y.item() if y.size == 1 else y[0])
        else:
            y_val = int(y)
        
        self.samples_seen += 1
        
        # Calculate prediction error using incremental classifier (DDM expects 0/1)
        prediction_error = self._calculate_incremental_prediction_error(x_dict, y_val)
        
        # Update detector based on backend
        drift_detected = False
        
        if self.backend == 'skmultiflow':
            drift_detected = self._update_skmultiflow(prediction_error)
        else:
            drift_detected = self._update_simple(prediction_error)
        
        # Update the incremental classifier after prediction
        try:
            self.incremental_classifier.learn_one(x_dict, y_val)
        except Exception as e:
            print(f"Warning: Classifier update error: {e}")
        
        if drift_detected:
            self.detections.append(self.time_step)
            self.detection_scores.append(prediction_error)
            # Reset classifier on drift detection
            self.incremental_classifier = tree.HoeffdingTreeClassifier()
        
        self.time_step += 1
        return drift_detected
    
    def _calculate_incremental_prediction_error(self, x_dict: dict, y_true: int) -> int:
        """Calculate prediction error using river incremental classifier (0 for correct, 1 for incorrect)"""
        
        if self.samples_seen < self.min_samples_for_prediction:
            return 0  # No error when insufficient data
        
        try:
            # Make prediction
            prediction = self.incremental_classifier.predict_one(x_dict)
            
            if prediction is None:
                return 0
            
            # Return error (1 for incorrect, 0 for correct)
            return 1 if prediction != y_true else 0
            
        except Exception as e:
            print(f"Warning: Prediction error: {e}")
            return 0
    
    def _update_skmultiflow(self, error: int) -> bool:
        """Update using skmultiflow DDM with correct API for version 0.5.3"""
        # skmultiflow 0.5.3 DDM API
        self.detector.add_element(error)
        drift_detected = self.detector.detected_change()
        return drift_detected
    
    def _update_simple(self, error: int) -> bool:
        """Simple DDM implementation as fallback"""
        self.sample_count += 1
        
        if error > 0:
            self.error_count += 1
        
        if self.sample_count < self.min_instances:
            return False
        
        # Calculate error rate and standard deviation
        self.p = self.error_count / self.sample_count
        self.s = np.sqrt(self.p * (1 - self.p) / self.sample_count)
        
        # Update minimum values
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s
        
        # Check for drift
        if self.p + self.s > self.p_min + self.s_min + self.drift_level * self.s_min:
            # Reset after drift detection
            self._reset_simple_ddm()
            return True
        elif self.p + self.s > self.p_min + self.s_min + self.warning_level * self.s_min:
            self.in_warning_zone = True
        else:
            self.in_warning_zone = False
        
        return False
    
    def _reset_simple_ddm(self):
        """Reset simple DDM state after drift detection"""
        self.error_count = 0
        self.sample_count = 0
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.p = 0.0
        self.s = 0.0
        self.in_warning_zone = False
    
    def reset(self):
        """Reset detector state"""
        if self.backend == 'skmultiflow' and SKMULTIFLOW_AVAILABLE:
            self.detector = DDM(warning_level=self.warning_level, out_control_level=self.drift_level)
        else:
            self._init_simple_ddm()
        
        self.detections = []
        self.time_step = 0
        self.detection_scores = []
        self.incremental_classifier = tree.HoeffdingTreeClassifier()
        self.samples_seen = 0