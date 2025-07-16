# data_streams/data_validator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import os
from .data_visualizer import DataStreamVisualizer

class DataStreamValidator:
    """Validator for synthetic data streams with concept drift"""
    
    def __init__(self):
        self.validation_results = {}
        self.visualizer = DataStreamVisualizer()
    
    def validate_data_stream(self, generator_type: str,
                           imbalance_ratio: List[float],
                           n_samples: int = 2000,
                           drift_start: int = 1000,
                           drift_end: int = 1500,
                           seed: int = 42) -> Dict[str, Any]:
        """Validate a single data stream configuration"""
        from data_streams.synthetic_generator import DataStreamFactory
        
        generator = DataStreamFactory.create_generator(
            generator_type=generator_type,
            n_samples=n_samples,
            drift_start=drift_start,
            drift_end=drift_end,
            drift_severity=0.15,
            drift_type='gradual',
            imbalance_ratio=imbalance_ratio,
            random_state=seed
        )
        
        df = generator.generate_stream()
        
        # Use the visualizer's validation method
        validation_result = self.visualizer.validate_data_quality(df, imbalance_ratio, generator_type)
        validation_result['data_frame'] = df
        
        return validation_result
    
    def plot_validation_results(self, validation_result: Dict[str, Any],
                               save_dir: str = "results/validation/") -> None:
        """Plot validation results using the visualizer"""
        df = validation_result['data_frame']
        generator_type = validation_result.get('generator_type', 'Unknown')
        expected_ratio = validation_result['expected_ratios']
        
        # Convert back to original ratio format
        ratio_sum = sum(expected_ratio)
        original_ratio = [int(r * ratio_sum) for r in expected_ratio]
        
        # Use visualizer's generate_validation_report method
        self.visualizer.generate_validation_report(
            df, original_ratio, generator_type, save_dir
        )
    
    def validate_all_configurations(self, save_dir: str = "results/validation/") -> pd.DataFrame:
        """Validate all experiment configurations"""
        configurations = [
            ('SINE1', [5, 5], 'Group A1'),
            ('SINE1', [7, 3], 'Group A2'),
            ('SINE1', [9, 1], 'Group A3'),
            ('SEA', [5, 5], 'Group B1'),
            ('SEA', [7, 3], 'Group B2'),
            ('SEA', [9, 1], 'Group B3')
        ]
        
        validation_results = []
        
        for generator_type, imbalance_ratio, group_name in configurations:
            print(f"Validating {group_name}: {generator_type} with ratio {imbalance_ratio}")
            
            result = self.validate_data_stream(
                generator_type=generator_type,
                imbalance_ratio=imbalance_ratio
            )
            
            result['group_name'] = group_name
            
            # Plot validation using the visualizer
            df = result['data_frame']
            self.visualizer.generate_validation_report(
                df, imbalance_ratio, generator_type, 
                os.path.join(save_dir, group_name)
            )
            
            # Store summary
            validation_results.append({
                'group_name': group_name,
                'generator_type': generator_type,
                'imbalance_ratio': f"{imbalance_ratio[0]}:{imbalance_ratio[1]}",
                'sample_count': result['total_samples'],
                'ratio_difference': result['ratio_difference'],
                'overall_quality': result['overall_quality']
            })
        
        summary_df = pd.DataFrame(validation_results)
        
        os.makedirs(save_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(save_dir, 'validation_summary.csv'), index=False)
        
        print("\nValidation Summary:")
        print(summary_df)
        
        # Check if all validations passed
        all_passed = True
        for _, row in summary_df.iterrows():
            if row['ratio_difference'] > 0.05:
                print(f"⚠️ Warning: {row['group_name']} has high ratio difference: {row['ratio_difference']:.3f}")
                all_passed = False
            # 移除了错误的 drift_detected 检查，因为该字段不存在
        
        if all_passed:
            print("\n✓ All data validations passed")
        else:
            print("\n⚠️ Some data validations have warnings, but continuing...")
        
        return summary_df