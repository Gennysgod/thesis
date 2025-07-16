import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import torch
from models.quadcdd_network import QuadCDDTrainer, create_quadcdd_model
from models.classifier import OnlineNaiveBayes
from .base_detector import BaseDetector
import os

class QuadCDDDetector(BaseDetector):
    """Simplified QuadCDD detector focusing on drift point detection"""
    
    def __init__(self, model_path: str = None, window_size: int = 100,
                 min_samples_before_detection: int = 200, **kwargs):
        super().__init__("QuadCDD")
        
        self.window_size = window_size
        self.min_samples_before_detection = min_samples_before_detection
        
        # Load pre-trained model
        self.model_path = model_path
        self.trainer = None
        self.load_pretrained_model()
        
        # Online classifier for accuracy sequence
        self.classifier = OnlineNaiveBayes()
        
        # Data storage
        self.accuracy_sequence = []
        
        # Detection state
        self.last_detection_time = -1000  # Prevent multiple detections too close
        self.min_detection_interval = 500
        
    def load_pretrained_model(self):
        """Load pre-trained QuadCDD model"""
        if self.model_path and os.path.exists(self.model_path):
            self.trainer = create_quadcdd_model()
            self.trainer.load_checkpoint(self.model_path)
            print(f"Loaded pre-trained QuadCDD model from {self.model_path}")
        else:
            print("Warning: No pre-trained model loaded. QuadCDD detector may not work properly.")
            self.trainer = create_quadcdd_model()
    
    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Update detector with new data point"""
        # Ensure inputs are properly formatted
        if np.isscalar(y):
            y = np.array([y])
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Update classifier and get accuracy
        accuracy = self.classifier.partial_fit(X[0], y[0])
        self.accuracy_sequence.append(accuracy)
        
        # Check if we have enough data
        if self.time_step < self.min_samples_before_detection:
            self.time_step += 1
            return False
        
        # Check if enough time has passed since last detection
        if self.time_step - self.last_detection_time < self.min_detection_interval:
            self.time_step += 1
            return False
        
        # Perform detection if we have enough accuracy history
        drift_detected = False
        if len(self.accuracy_sequence) >= self.window_size:
            drift_detected = self._detect_drift()
        
        self.time_step += 1
        return drift_detected
    
    def _detect_drift(self) -> bool:
        """Detect drift using QuadCDD model"""
        if self.trainer is None:
            return False
        
        # Get recent accuracy window
        recent_accuracy = np.array(self.accuracy_sequence[-self.window_size:])
        
        try:
            # Predict quadruple
            quadruple = self.trainer.predict(recent_accuracy)
            
            # Extract normalized drift start (Ds)
            ds_normalized = quadruple['Ds']
            
            # Convert to actual detection point
            # Ds represents position within the window
            window_start = max(0, self.time_step - self.window_size)
            detection_point = window_start + int(ds_normalized * self.window_size)
            
            # Check if this indicates a drift
            # Ds > 0.3 suggests drift is happening in the window
            if ds_normalized > 0.3 and quadruple['Dv'] > 0.1:
                self.detections.append(self.time_step)
                self.last_detection_time = self.time_step
                self.detection_scores.append(quadruple['Dv'])
                return True
                
        except Exception as e:
            print(f"Error in QuadCDD prediction: {e}")
            
        return False
    
    def reset(self):
        """Reset detector state"""
        super().reset()
        self.classifier.reset()
        self.accuracy_sequence = []
        self.last_detection_time = -1000