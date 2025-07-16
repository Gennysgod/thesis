import numpy as np
from .base_detector import BaseDetector
from collections import deque
from river import tree

# Try to import skmultiflow ADWIN
try:
    from skmultiflow.drift_detection import ADWIN
    SKMULTIFLOW_AVAILABLE = True
    print("✓ skmultiflow ADWIN imported successfully")
except ImportError:
    SKMULTIFLOW_AVAILABLE = False
    print("✗ skmultiflow ADWIN not available, using simple fallback")

class ADWINDetector(BaseDetector):
    """ADWIN drift detector with river-based incremental learning"""
    
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
        
        # River-based incremental classifier for prediction
        self.incremental_classifier = tree.HoeffdingTreeClassifier()
        self.samples_seen = 0
        self.min_samples_for_prediction = 10
        
    def _init_simple_adwin(self):
        """Initialize simple ADWIN implementation as fallback"""
        self.detector = None
        self.backend = 'simple'
        self.error_window = deque(maxlen=1000)
        self.min_window_size = 30
        self.warning_threshold = 0.1
        print("✓ ADWIN initialized with simple fallback backend")
        
    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Update ADWIN with new data point using river-based prediction"""
        
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
        
        # Calculate prediction error using incremental classifier
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
    
    def _calculate_incremental_prediction_error(self, x_dict: dict, y_true: int) -> float:
        """Calculate prediction error using river incremental classifier"""
        
        if self.samples_seen < self.min_samples_for_prediction:
            return 0.0  # No error when insufficient data
        
        try:
            # Make prediction
            prediction = self.incremental_classifier.predict_one(x_dict)
            
            if prediction is None:
                return 0.0
            
            # Return error (1.0 for incorrect, 0.0 for correct)
            return 1.0 if prediction != y_true else 0.0
            
        except Exception as e:
            print(f"Warning: Prediction error: {e}")
            return 0.0
    
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
        self.incremental_classifier = tree.HoeffdingTreeClassifier()
        self.samples_seen = 0