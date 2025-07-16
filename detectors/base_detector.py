from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class BaseDetector(ABC):
    """Base class for all concept drift detectors"""
    
    def __init__(self, name: str):
        self.name = name
        self.detections = []  # List of detection time points
        self.time_step = 0
        self.is_detecting = False
        self.detection_scores = []  # Store detection scores/statistics
        
    @abstractmethod
    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Update detector with new data point and return if drift detected
        
        Args:
            X: Feature vector (can be single sample or batch)
            y: Target value (can be single value or batch)
            
        Returns:
            bool: True if drift detected at this time step
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset detector to initial state"""
        pass
    
    def get_detections(self) -> List[int]:
        """Get list of detection time points"""
        return self.detections.copy()
    
    def get_detection_count(self) -> int:
        """Get total number of detections"""
        return len(self.detections)
    
    def get_last_detection(self) -> Optional[int]:
        """Get last detection time point"""
        return self.detections[-1] if self.detections else None
    
    def get_detection_scores(self) -> List[float]:
        """Get detection scores over time"""
        return self.detection_scores.copy()