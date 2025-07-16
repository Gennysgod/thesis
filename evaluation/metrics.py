import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DriftDetectionMetrics:
    """Container for drift detection evaluation metrics"""
    drift_detection_delay: float
    false_positive_rate: float
    missed_drift_count: int
    drift_detection_recall: float
    # Additional statistics
    total_detections: int
    true_positives: int
    false_positives: int
    detection_times: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'drift_detection_delay': self.drift_detection_delay,
            'false_positive_rate': self.false_positive_rate,
            'missed_drift_count': self.missed_drift_count,
            'drift_detection_recall': self.drift_detection_recall,
            'total_detections': self.total_detections,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives
        }

def calculate_single_run_metrics(detected_drifts: List[int],
                                true_drift_start: int,
                                true_drift_end: int,
                                stream_length: int,
                                tolerance_window: int = 300) -> DriftDetectionMetrics:
    """
    Calculate metrics for a single experimental run
    
    Args:
        detected_drifts: List of detection time points
        true_drift_start: True drift start time
        true_drift_end: True drift end time  
        stream_length: Total stream length
        tolerance_window: Tolerance window for detection
        
    Returns:
        DriftDetectionMetrics object
    """
    evaluator = DriftEvaluator(tolerance_window)
    return evaluator.evaluate_detection(
        detected_drifts=detected_drifts,
        true_drift_points=[true_drift_start],
        stream_length=stream_length
    )

class DriftEvaluator:
    """Evaluator for concept drift detection performance"""
    
    def __init__(self, tolerance_window: int = 300):
        """
        Initialize evaluator
        Args:
            tolerance_window: Extended window for detection tolerance
        """
        self.tolerance_window = tolerance_window
    
    def evaluate_detection(self,
                          detected_drifts: List[int],
                          true_drift_points: List[int],
                          stream_length: int,
                          drift_active_periods: List[Tuple[int, int]] = None) -> DriftDetectionMetrics:
        """
        Evaluate drift detection performance with improved metrics
        """
        # Convert to numpy arrays for easier computation
        detected_drifts = np.array(detected_drifts) if detected_drifts else np.array([])
        true_drift_points = np.array(true_drift_points) if true_drift_points else np.array([])
        
        # Calculate metrics with improved logic
        detection_delay = self._calculate_detection_delay_improved(detected_drifts, true_drift_points)
        fpr = self._calculate_false_positive_rate_improved(detected_drifts, true_drift_points, stream_length)
        missed_count = self._calculate_missed_drift_count_improved(detected_drifts, true_drift_points)
        recall = self._calculate_drift_detection_recall_improved(detected_drifts, true_drift_points)
        
        # Count true/false positives
        tp, fp = self._classify_detections_improved(detected_drifts, true_drift_points)
        
        return DriftDetectionMetrics(
            drift_detection_delay=detection_delay,
            false_positive_rate=fpr,
            missed_drift_count=missed_count,
            drift_detection_recall=recall,
            total_detections=len(detected_drifts),
            true_positives=tp,
            false_positives=fp,
            detection_times=detected_drifts.tolist()
        )
    
    def _calculate_detection_delay_improved(self, detected_drifts: np.ndarray,
                                          true_drift_points: np.ndarray) -> float:
        """Calculate detection delay with improved logic"""
        if len(true_drift_points) == 0:
            return 0.0
            
        true_drift_start = true_drift_points[0]  # Should be 1000
        
        # Find detections after drift start
        valid_detections = detected_drifts[detected_drifts >= true_drift_start]
        
        if len(valid_detections) > 0:
            # Return delay of first valid detection
            return float(valid_detections[0] - true_drift_start)
        else:
            # No valid detection, return maximum delay
            return 1000.0  # Maximum reasonable delay
    
    def _calculate_false_positive_rate_improved(self, detected_drifts: np.ndarray,
                                               true_drift_points: np.ndarray,
                                               stream_length: int) -> float:
        """Calculate FPR with improved window"""
        if len(detected_drifts) == 0:
            return 0.0
        
        true_drift_start = true_drift_points[0] if len(true_drift_points) > 0 else 1000
        
        # Define valid detection window: [drift_start - 200, drift_start + 700]
        window_start = max(0, true_drift_start - 200)
        window_end = min(stream_length, true_drift_start + 700)
        
        # Count detections outside valid window as false positives
        false_positives = 0
        for detection in detected_drifts:
            if detection < window_start or detection > window_end:
                false_positives += 1
        
        return false_positives / len(detected_drifts) if len(detected_drifts) > 0 else 0.0
    
    def _calculate_missed_drift_count_improved(self, detected_drifts: np.ndarray,
                                              true_drift_points: np.ndarray) -> int:
        """Calculate missed drift with improved window"""
        if len(true_drift_points) == 0:
            return 0
        
        true_drift_start = true_drift_points[0]
        
        # Check if any detection in extended window [drift_start - 200, drift_start + 600]
        window_start = true_drift_start - 200
        window_end = true_drift_start + 600
        
        detections_in_window = detected_drifts[
            (detected_drifts >= window_start) & (detected_drifts <= window_end)
        ]
        
        # If we have any detection in the window, drift was not missed
        return 0 if len(detections_in_window) > 0 else 1
    
    def _calculate_drift_detection_recall_improved(self, detected_drifts: np.ndarray,
                                                  true_drift_points: np.ndarray) -> float:
        """Calculate recall based on missed drift count"""
        if len(true_drift_points) == 0:
            return 1.0
        
        missed = self._calculate_missed_drift_count_improved(detected_drifts, true_drift_points)
        return 1.0 - missed  # If not missed, recall is 1.0
    
    def _classify_detections_improved(self, detected_drifts: np.ndarray,
                                     true_drift_points: np.ndarray) -> Tuple[int, int]:
        """Classify detections as true positives or false positives"""
        if len(true_drift_points) == 0:
            return 0, len(detected_drifts)
        
        true_drift_start = true_drift_points[0]
        window_start = true_drift_start - 200
        window_end = true_drift_start + 700
        
        true_positives = 0
        false_positives = 0
        
        for detection in detected_drifts:
            if window_start <= detection <= window_end:
                true_positives += 1
            else:
                false_positives += 1
        
        return true_positives, false_positives

class ExperimentEvaluator:
    """Evaluator for multiple experimental runs"""
    
    def __init__(self, tolerance_window: int = 300):
        self.drift_evaluator = DriftEvaluator(tolerance_window)
        self.results_history = []
    
    def evaluate_experiment_run(self, detector_name: str, data_group: str,
                               detected_drifts: List[int], true_drift_points: List[int],
                               stream_length: int, run_id: int,
                               drift_active_periods: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Evaluate single experiment run"""
        metrics = self.drift_evaluator.evaluate_detection(
            detected_drifts, true_drift_points, stream_length, drift_active_periods
        )
        
        result = {
            'detector': detector_name,
            'data_group': data_group,
            'run_id': run_id,
            'stream_length': stream_length,
            **metrics.to_dict()
        }
        
        self.results_history.append(result)
        return result
    
    def aggregate_results(self, group_by: List[str] = ['detector', 'data_group']) -> pd.DataFrame:
        """Aggregate results across multiple runs"""
        if not self.results_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results_history)
        
        # Define aggregation functions with consistent naming
        agg_functions = {
            'drift_detection_delay': ['mean', 'std', 'median'],
            'false_positive_rate': ['mean', 'std'],
            'missed_drift_count': ['mean', 'std', 'sum'],
            'drift_detection_recall': ['mean', 'std'],
            'total_detections': ['mean', 'std'],
            'true_positives': ['mean', 'std'],
            'false_positives': ['mean', 'std']
        }
        
        # Group and aggregate
        grouped = df.groupby(group_by)
        aggregated = grouped.agg(agg_functions)
        
        # Flatten column names with underscores
        aggregated.columns = [f'{col[0]}_{col[1]}' for col in aggregated.columns]
        
        # Add count of runs
        aggregated['num_runs'] = grouped.size()
        
        return aggregated.reset_index()
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all results"""
        if not self.results_history:
            return {}
        
        df = pd.DataFrame(self.results_history)
        
        summary = {
            'total_runs': len(df),
            'unique_detectors': df['detector'].nunique(),
            'unique_data_groups': df['data_group'].nunique(),
            'overall_metrics': {
                'avg_delay': df['drift_detection_delay'].mean(),
                'avg_fpr': df['false_positive_rate'].mean(),
                'avg_recall': df['drift_detection_recall'].mean(),
                'total_missed_drifts': df['missed_drift_count'].sum()
            },
            'detector_performance': df.groupby('detector').agg({
                'drift_detection_delay': 'mean',
                'false_positive_rate': 'mean',
                'drift_detection_recall': 'mean',
                'missed_drift_count': 'mean'
            }).to_dict('index'),
            'data_group_performance': df.groupby('data_group').agg({
                'drift_detection_delay': 'mean',
                'false_positive_rate': 'mean',
                'drift_detection_recall': 'mean',
                'missed_drift_count': 'mean'
            }).to_dict('index')
        }
        
        return summary