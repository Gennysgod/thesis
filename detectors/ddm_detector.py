import numpy as np
from .base_detector import BaseDetector
from collections import deque

# Try to import skmultiflow DDM
try:
    from skmultiflow.drift_detection import DDM
    SKMULTIFLOW_AVAILABLE = True
    print("✓ skmultiflow DDM imported successfully")
except ImportError:
    SKMULTIFLOW_AVAILABLE = False
    print("✗ skmultiflow DDM not available, using simple fallback")

class DDMDetector(BaseDetector):
    """DDM drift detector with skmultiflow support and fallback"""
    
    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0, **kwargs):
        super().__init__("DDM")
        self.warning_level = warning_level
        self.drift_level = drift_level
        
        # Initialize DDM detector
        if SKMULTIFLOW_AVAILABLE:
            # skmultiflow 0.5.3 DDM uses 'out_control_level' instead of 'drift_level'
            self.detector = DDM(warning_level=warning_level, out_control_level=drift_level)
            self.backend = 'skmultiflow'
            print(f"✓ DDM initialized with skmultiflow backend (warn={warning_level}, out_control={drift_level})")
        else:
            self._init_simple_ddm()
        
        # Error tracking for prediction
        self.window_size = 30
        self.feature_window = deque(maxlen=self.window_size)
        self.label_window = deque(maxlen=self.window_size)
        
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
        """Update DDM with new data point"""
        # Store recent data for prediction
        self.feature_window.append(X)
        self.label_window.append(y)
        
        # Calculate prediction error (DDM expects 0 for correct, 1 for incorrect)
        prediction_error = self._calculate_prediction_error(y)
        
        # Update detector based on backend
        drift_detected = False
        
        if self.backend == 'skmultiflow':
            drift_detected = self._update_skmultiflow(prediction_error)
        else:
            drift_detected = self._update_simple(prediction_error)
        
        if drift_detected:
            self.detections.append(self.time_step)
            self.detection_scores.append(prediction_error)
        
        self.time_step += 1
        return drift_detected
    
    def _calculate_prediction_error(self, y: np.ndarray) -> int:
        """Calculate prediction error (0 for correct, 1 for incorrect prediction)"""
        if len(self.label_window) < 10:
            return 0  # No error when insufficient data
        
        # Use majority class of recent window as prediction
        recent_labels = list(self.label_window)[-10:]
        majority_class = 1 if np.mean(recent_labels) > 0.5 else 0
        
        # Convert y to scalar if needed
        if np.isarray(y):
            y_val = y.item() if y.size == 1 else y[0]
        else:
            y_val = y
        
        # DDM expects 1 for error, 0 for correct prediction
        prediction_error = 1 if y_val != majority_class else 0
        return prediction_error
    
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
        self.feature_window.clear()
        self.label_window.clear()