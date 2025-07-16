import numpy as np
from .base_detector import BaseDetector

class SimpleDriftDetector(BaseDetector):
    """Simple drift detector based on accuracy monitoring"""

    def __init__(self, window_size: int = 100, threshold: float = 0.1, name: str = "Simple"):
        super().__init__(name)
        self.window_size = window_size
        self.threshold = threshold
        self.accuracy_history = []
        self.baseline_accuracy = None
        self.recent_labels = []

    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Update detector with new data point"""
        # Ensure y is scalar
        if np.isarray(y):
            y = y.item() if y.size == 1 else y[0]

        # Simple prediction using majority class
        if len(self.recent_labels) > 10:
            majority_class = round(np.mean(self.recent_labels[-10:]))
            accuracy = 1.0 if y == majority_class else 0.0
        else:
            accuracy = 1.0  # Perfect accuracy when insufficient history

        # Update recent labels
        self.recent_labels.append(y)
        if len(self.recent_labels) > self.window_size:
            self.recent_labels = self.recent_labels[-self.window_size:]

        # Update accuracy history
        self.accuracy_history.append(accuracy)
        if len(self.accuracy_history) > self.window_size * 2:
            self.accuracy_history = self.accuracy_history[-self.window_size * 2:]

        # Set baseline if we have enough data
        if self.baseline_accuracy is None and len(self.accuracy_history) >= self.window_size:
            self.baseline_accuracy = np.mean(self.accuracy_history[:self.window_size])

        # Detect drift
        drift_detected = False
        if (self.baseline_accuracy is not None and 
            len(self.accuracy_history) >= self.window_size):
            
            recent_accuracy = np.mean(self.accuracy_history[-self.window_size//2:])
            accuracy_drop = self.baseline_accuracy - recent_accuracy
            
            if accuracy_drop > self.threshold:
                drift_detected = True
                # Update baseline
                self.baseline_accuracy = recent_accuracy

        if drift_detected:
            self.detections.append(self.time_step)

        self.detection_scores.append(accuracy)
        self.time_step += 1
        return drift_detected

    def reset(self):
        """Reset detector state"""
        super().reset()
        self.accuracy_history = []
        self.baseline_accuracy = None
        self.recent_labels = []


class SimpleADWIN(SimpleDriftDetector):
    """Simple ADWIN-like detector"""
    
    def __init__(self, **kwargs):
        super().__init__(name="SimpleADWIN", **kwargs)


class SimpleDDM(SimpleDriftDetector):
    """Simple DDM-like detector"""
    
    def __init__(self, **kwargs):
        super().__init__(name="SimpleDDM", threshold=0.15, **kwargs)

