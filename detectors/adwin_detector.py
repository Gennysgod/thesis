import numpy as np
from .base_detector import BaseDetector
from collections import deque

# Try to import skmultiflow ADWIN
try:
    from skmultiflow.drift_detection import ADWIN
    SKMULTIFLOW_AVAILABLE = True
    print("✓ skmultiflow ADWIN imported successfully")
except ImportError:
    SKMULTIFLOW_AVAILABLE = False
    print("✗ skmultiflow ADWIN not available, using simple fallback")

class ADWINDetector(BaseDetector):
    """ADWIN drift detector with skmultiflow support and fallback"""
    
    def __init__(self, delta: float = 0.002, **kwargs):
        super().__init__("ADWIN")
        self.delta = delta
        
        # Initialize ADWIN detector
        if SKMULTIFLOW_AVAILABLE:
            try:
                self.detector = ADWIN(delta=delta)
                self.backend = 'skmultiflow'
                print(f"✓ ADWIN initialized with skmultiflow backend (delta={delta})")
            except Exception as e:
                print(f"✗ skmultiflow ADWIN initialization failed: {e}")
                self._init_simple_adwin()
        else:
            self._init_simple_adwin()
        
        # Error tracking for prediction
        self.window_size = 30
        self.feature_window = deque(maxlen=self.window_size)
        self.label_window = deque(maxlen=self.window_size)
        
    def _init_simple_adwin(self):
        """Initialize simple ADWIN implementation as fallback"""
        self.detector = None
        self.backend = 'simple'
        self.error_window = deque(maxlen=1000)
        self.min_window_size = 30
        self.warning_threshold = 0.1
        print("✓ ADWIN initialized with simple fallback backend")
        
    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Update ADWIN with new data point"""
        # Store recent data for prediction
        self.feature_window.append(X)
        self.label_window.append(y)
        
        # Calculate prediction error
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
    
    def _calculate_prediction_error(self, y: np.ndarray) -> float:
        """Calculate prediction error using simple majority class prediction"""
        if len(self.label_window) < 10:
            return 0.0
        
        # Use majority class of recent window as prediction
        recent_labels = list(self.label_window)[-10:]
        majority_class = 1 if np.mean(recent_labels) > 0.5 else 0
        
        # Convert y to scalar if needed
        if np.isarray(y):
            y_val = y.item() if y.size == 1 else y[0]
        else:
            y_val = y
        
        prediction_error = 1.0 if y_val != majority_class else 0.0
        return prediction_error
    
    def _update_skmultiflow(self, error: float) -> bool:
        """Update using skmultiflow ADWIN with correct API for version 0.5.3"""
        # skmultiflow 0.5.3 ADWIN API
        self.detector.add_element(error)
        drift_detected = self.detector.detected_change()
        return drift_detected
    
    def _update_simple(self, error: float) -> bool:
        """Simple ADWIN-like implementation as fallback"""
        self.error_window.append(error)
        
        if len(self.error_window) < self.min_window_size:
            return False
        
        # Simple drift detection: compare recent vs historical error rates
        window_size = len(self.error_window)
        if window_size < 50:
            return False
        
        # Split window into two parts
        split_point = window_size // 2
        recent_errors = list(self.error_window)[split_point:]
        historical_errors = list(self.error_window)[:split_point]
        
        recent_rate = np.mean(recent_errors)
        historical_rate = np.mean(historical_errors)
        
        # Detect significant increase in error rate
        if recent_rate - historical_rate > self.warning_threshold:
            return True
        
        return False
    
    def reset(self):
        """Reset detector state"""
        if self.backend == 'skmultiflow' and SKMULTIFLOW_AVAILABLE:
            self.detector = ADWIN(delta=self.delta)
        else:
            self._init_simple_adwin()
        
        self.detections = []
        self.time_step = 0
        self.detection_scores = []
        self.feature_window.clear()
        self.label_window.clear()