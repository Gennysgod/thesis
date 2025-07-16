# quick_test.py

"""
Quick test script to run experiment without QuadCDD (which requires pre-training)
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import time
import fix_numpy_float

def run_quick_experiment():
    """Run a quick experiment with simplified parameters"""
    print("=" * 60)
    print("RUNNING QUICK EXPERIMENT TEST")
    print("=" * 60)
    
    # Import necessary modules
    from data_streams.synthetic_generator import DataStreamFactory
    from detectors.adwin_detector import ADWINDetector
    from detectors.ddm_detector import DDMDetector
    from evaluation.metrics import calculate_single_run_metrics
    
    # Test configuration
    config = {
        'data_groups': {
            'Group A1': {'attributes': 2, 'generator': 'SINE1', 'imbalance_ratio': [5, 5]},
            'Group A2': {'attributes': 2, 'generator': 'SINE1', 'imbalance_ratio': [7, 3]}
        },
        'detectors': ['ADWIN', 'DDM'],  # Skip QuadCDD for now
        'n_runs': 2,  # Reduced for quick testing
        'stream_length': 1000,  # Shorter streams
        'drift_params': {
            'Ds': 400,  # Earlier drift start
            'De': 600,   # Earlier drift end
            'Dv': 0.2,
            'Dt': 'gradual'
        }
    }
    
    results = []
    
    print(f"Testing {len(config['detectors'])} detectors on {len(config['data_groups'])} data groups")
    print(f"Running {config['n_runs']} runs per combination")
    
    experiment_id = 0
    
    for detector_name in config['detectors']:
        for group_name, group_config in config['data_groups'].items():
            for run_id in range(config['n_runs']):
                
                print(f"\nExperiment {experiment_id + 1}: {detector_name} on {group_name}, Run {run_id + 1}")
                
                try:
                    # Generate data stream
                    print("  Generating data stream...")
                    generator = DataStreamFactory.create_generator(
                        group_config['generator'],
                        n_samples=config['stream_length'],
                        drift_start=config['drift_params']['Ds'],
                        drift_end=config['drift_params']['De'],
                        drift_severity=config['drift_params']['Dv'],
                        drift_type=config['drift_params']['Dt'],
                        imbalance_ratio=group_config['imbalance_ratio'],
                        random_state=42 + experiment_id
                    )
                    
                    data_stream = generator.generate_stream()
                    print(f"  Generated {len(data_stream)} samples")
                    
                    # Create detector
                    print(f"  Creating {detector_name} detector...")
                    if detector_name == 'ADWIN':
                        detector = ADWINDetector(delta=0.002)
                    elif detector_name == 'DDM':
                        detector = DDMDetector(warning_level=2.0, drift_level=3.0)
                    else:
                        raise ValueError(f"Unknown detector: {detector_name}")
                    
                    # Process data stream
                    print("  Processing data stream...")
                    detected_drifts = []
                    
                    for i, row in data_stream.iterrows():
                        # Extract features and label
                        feature_cols = [col for col in data_stream.columns if col.startswith('x')]
                        features = row[feature_cols].values
                        label = row['y']
                        
                        # Update detector
                        drift_detected = detector.update(features, label)
                        if drift_detected:
                            detected_drifts.append(i)
                    
                    print(f"  Detected {len(detected_drifts)} drifts at times: {detected_drifts}")
                    
                    # Calculate metrics
                    print("  Calculating metrics...")
                    metrics = calculate_single_run_metrics(
                        detected_drifts=detected_drifts,
                        true_drift_start=config['drift_params']['Ds'],
                        true_drift_end=config['drift_params']['De'],
                        stream_length=config['stream_length'],
                        tolerance_window=50
                    )
                    
                    # Store results
                    result = {
                        'detector': detector_name,
                        'data_group': group_name,
                        'run_id': run_id,
                        'experiment_id': experiment_id,
                        'detected_drifts': detected_drifts,
                        'n_detections': len(detected_drifts),
                        'drift_detection_delay': metrics.drift_detection_delay,
                        'false_positive_rate': metrics.false_positive_rate,
                        'missed_drift_count': metrics.missed_drift_count,
                        'drift_detection_recall': metrics.drift_detection_recall,
                        'success': True
                    }
                    
                    results.append(result)
                    
                    print(f"  Results: Delay={metrics.drift_detection_delay:.1f}, "
                          f"FPR={metrics.false_positive_rate:.3f}, "
                          f"Recall={metrics.drift_detection_recall:.3f}")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    result = {
                        'detector': detector_name,
                        'data_group': group_name,
                        'run_id': run_id,
                        'experiment_id': experiment_id,
                        'error': str(e),
                        'success': False
                    }
                    results.append(result)
                
                experiment_id += 1
    
    # Summarize results
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if successful_results:
        print("\nSuccessful experiments:")
        for result in successful_results:
            print(f"  {result['detector']} on {result['data_group']}: "
                  f"Delay={result['drift_detection_delay']:.1f}, "
                  f"Detections={result['n_detections']}")
    
    if failed_results:
        print("\nFailed experiments:")
        for result in failed_results:
            print(f"  {result['detector']} on {result['data_group']}: {result.get('error', 'Unknown error')}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        output_file = 'quick_test_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    success_rate = len(successful_results) / len(results) if results else 0
    print(f"\nOverall success rate: {success_rate:.1%}")
    
    return success_rate > 0.5

def main():
    """Main function"""
    try:
        success = run_quick_experiment()
        if success:
            print("\n✓ Quick experiment completed successfully!")
            print("You can now try running the full experiment with:")
            print("  python main.py --quick-test")
        else:
            print("\n✗ Quick experiment had issues. Please check the errors above.")
        return success
    except Exception as e:
        print(f"\n✗ Quick experiment failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)