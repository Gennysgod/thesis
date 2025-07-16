# experiments/run_experiment.py
"""
Main experiment execution module.
Runs the complete experimental pipeline for concept drift detection evaluation.
"""
import numpy as np
import pandas as pd
import time
import os
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import logging
from datetime import datetime

# Import project modules
from data_streams.synthetic_generator import DataStreamFactory
from detectors.adwin_detector import ADWINDetector
from detectors.ddm_detector import DDMDetector
from detectors.quadcdd_detector import QuadCDDDetector
from evaluation.metrics import ExperimentEvaluator, calculate_single_run_metrics
from evaluation.visualizer import ExperimentVisualizer
from experiments.experiment_config import get_config_manager, validate_experiment_setup
from utils.common import setup_logging, ensure_directory, save_results

class ExperimentRunner:
    """Main class for running concept drift detection experiments"""
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config_manager = get_config_manager(config_override)
        self.logger = setup_logging(
            os.path.join(
                self.config_manager.paths['logs_dir'], 
                self.config_manager.output_files['experiment_log']
            )
        )
        
        # Create output directories
        self.config_manager.create_output_directories()
        
        # Initialize evaluator and visualizer
        tolerance_window = self.config_manager.config['detection_params']['tolerance_window']
        self.evaluator = ExperimentEvaluator(tolerance_window)
        self.visualizer = ExperimentVisualizer()
        
        # Results storage
        self.all_results = []
        self.experiment_start_time = None
        
    def create_detector(self, detector_name: str, detector_config: Dict[str, Any]):
        """Create detector instance based on configuration"""
        
        if detector_name == 'ADWIN':
            return ADWINDetector(
                delta=detector_config.get('delta', 0.002)
            )
        elif detector_name == 'DDM':
            return DDMDetector(
                warning_level=detector_config.get('warning_level', 2.0),
                drift_level=detector_config.get('drift_level', 3.0)
            )
        elif detector_name == 'QuadCDD':
            return QuadCDDDetector(
                model_path=detector_config.get('model_path'),
                window_size=detector_config.get('window_size', 100),
                min_detection_window=detector_config.get('min_detection_window', 200),
                detection_threshold=detector_config.get('detection_threshold', 0.2)
            )
        else:
            raise ValueError(f"Unknown detector: {detector_name}")
    
    def generate_data_stream(self, group_config: Dict[str, Any], seed: int) -> pd.DataFrame:
        """Generate data stream for experiment"""
        
        # Get drift parameters
        drift_params = self.config_manager.config['drift_params']
        
        # Create generator
        generator = DataStreamFactory.create_generator(
            generator_type=group_config['generator'],
            n_samples=self.config_manager.config['stream_length'],
            drift_start=drift_params['Ds'],
            drift_end=drift_params['De'],
            drift_severity=drift_params['Dv'],
            drift_type=drift_params['Dt'],
            imbalance_ratio=group_config['imbalance_ratio'],
            random_state=seed
        )
        
        return generator.generate_stream()
    
    def run_single_experiment(self, combination: Dict[str, Any]) -> Dict[str, Any]:
        """Run single experiment combination"""
        
        detector_name = combination['detector']
        data_group = combination['data_group']
        run_id = combination['run_id']
        seed = combination['seed']
        
        self.logger.info(f"Running: {detector_name} on {data_group}, Run {run_id+1}")
        
        try:
            # Generate data stream
            data_stream = self.generate_data_stream(combination['group_config'], seed)
            
            # Create detector
            detector = self.create_detector(detector_name, combination['detector_config'])
            
            # Process data stream
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
            
            # Calculate metrics
            drift_params = self.config_manager.config['drift_params']
            tolerance_window = self.config_manager.config['detection_params']['tolerance_window']
            
            metrics = calculate_single_run_metrics(
                detected_drifts=detected_drifts,
                true_drift_start=drift_params['Ds'],
                true_drift_end=drift_params['De'],
                stream_length=self.config_manager.config['stream_length'],
                tolerance_window=tolerance_window
            )
            
            # Create result record
            result = {
                'detector': detector_name,
                'data_group': data_group,
                'run_id': run_id,
                'seed': seed,
                'detected_drifts': detected_drifts,
                'n_detections': len(detected_drifts),
                'drift_detection_delay': metrics.drift_detection_delay,
                'false_positive_rate': metrics.false_positive_rate,
                'missed_drift_count': metrics.missed_drift_count,
                'drift_detection_recall': metrics.drift_detection_recall,
                'true_positives': metrics.true_positives,
                'false_positives': metrics.false_positives,
                'total_detections': metrics.total_detections,
                'stream_length': len(data_stream),
                'imbalance_ratio': combination['group_config']['imbalance_ratio'],
                'generator_type': combination['group_config']['generator'],
                'execution_time': time.time(),
                'success': True
            }
            
            # Add detector-specific information
            if hasattr(detector, 'get_detection_statistics'):
                detector_stats = detector.get_detection_statistics()
                result['detector_stats'] = detector_stats
            
            self.logger.info(f"Completed: {detector_name} on {data_group}, Run {run_id+1} - "
                           f"Detections: {len(detected_drifts)}, "
                           f"Delay: {metrics.drift_detection_delay:.1f}, "
                           f"FPR: {metrics.false_positive_rate:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {detector_name} on {data_group}, Run {run_id+1}: {str(e)}")
            
            # Return error result
            return {
                'detector': detector_name,
                'data_group': data_group,
                'run_id': run_id,
                'seed': seed,
                'error': str(e),
                'success': False
            }
    
    def run_all_experiments(self) -> pd.DataFrame:
        """Run all experiment combinations"""
        
        self.experiment_start_time = time.time()
        self.logger.info("Starting complete experimental pipeline")
        
        # Get all combinations
        combinations = self.config_manager.get_experiment_combinations()
        total_experiments = len(combinations)
        
        self.logger.info(f"Total experiments to run: {total_experiments}")
        self.config_manager.print_summary()
        
        # Run experiments with progress bar
        results = []
        failed_experiments = 0
        
        with tqdm(total=total_experiments, desc="Running experiments") as pbar:
            for combination in combinations:
                result = self.run_single_experiment(combination)
                results.append(result)
                
                if not result.get('success', False):
                    failed_experiments += 1
                
                # Update progress
                pbar.set_postfix({
                    'Detector': combination['detector'],
                    'Group': combination['data_group'],
                    'Failed': failed_experiments
                })
                pbar.update(1)
                
                # Save intermediate results
                if self.config_manager.exec_config.get('save_intermediate', True):
                    if len(results) % 50 == 0:  # Save every 50 experiments
                        self._save_intermediate_results(results)
        
        self.all_results = results
        
        # Log completion statistics
        successful_experiments = total_experiments - failed_experiments
        total_time = time.time() - self.experiment_start_time
        
        self.logger.info(f"Experiments completed: {successful_experiments}/{total_experiments}")
        self.logger.info(f"Failed experiments: {failed_experiments}")
        self.logger.info(f"Total execution time: {total_time/3600:.2f} hours")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        return results_df
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]):
        """Save intermediate results"""
        intermediate_file = os.path.join(
            self.config_manager.paths['raw_results_dir'],
            f"intermediate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        df = pd.DataFrame(results)
        df.to_csv(intermediate_file, index=False)
    
    def aggregate_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate results across multiple runs"""
        
        self.logger.info("Aggregating experimental results")
        
        # Filter successful experiments only
        successful_results = results_df[results_df['success'] == True].copy()
        
        if len(successful_results) == 0:
            self.logger.error("No successful experiments to aggregate")
            return pd.DataFrame()
        
        # Define aggregation functions
        agg_functions = {
            'drift_detection_delay': ['mean', 'std', 'median', 'min', 'max'],
            'false_positive_rate': ['mean', 'std', 'count'],
            'missed_drift_count': ['mean', 'std', 'sum'],
            'drift_detection_recall': ['mean', 'std', 'median'],
            'n_detections': ['mean', 'std'],
            'true_positives': ['mean', 'std'],
            'false_positives': ['mean', 'std']
        }
        
        # Group by detector and data group
        grouped = successful_results.groupby(['detector', 'data_group'])
        aggregated = grouped.agg(agg_functions)
        
        # Flatten column names
        aggregated.columns = [f'{col[0]}_{col[1]}' for col in aggregated.columns]
        
        # Add additional statistics
        aggregated['num_runs'] = grouped.size()
        aggregated['success_rate'] = grouped.size() / self.config_manager.config['n_runs']
        
        # Add imbalance ratio information
        def get_imbalance_ratio(group_name):
            group_config = self.config_manager.get_data_group_config(group_name)
            ratio = group_config['imbalance_ratio']
            return f"{ratio[0]}:{ratio[1]}"
        
        aggregated = aggregated.reset_index()
        aggregated['imbalance_ratio'] = aggregated['data_group'].apply(get_imbalance_ratio)
        
        self.logger.info(f"Aggregated results for {len(aggregated)} combinations")
        
        return aggregated
    
    def evaluate_quality(self, aggregated_results: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate experiment quality against thresholds"""
        
        self.logger.info("Evaluating experiment quality")
        
        thresholds = self.config_manager.config['quality_thresholds']
        
        quality_results = {
            'thresholds': thresholds,
            'overall_quality': True,
            'detector_quality': {},
            'failed_combinations': []
        }
        
        for _, row in aggregated_results.iterrows():
            detector = row['detector']
            data_group = row['data_group']
            
            # Check thresholds
            avg_delay = row.get('drift_detection_delay_mean', float('inf'))
            avg_fpr = row.get('false_positive_rate_mean', 1.0)
            avg_recall = row.get('drift_detection_recall_mean', 0.0)
            
            meets_delay = avg_delay <= thresholds['delay_threshold']
            meets_fpr = avg_fpr <= thresholds['fpr_threshold']
            meets_recall = avg_recall >= thresholds['min_recall']
            
            combination_quality = meets_delay and meets_fpr and meets_recall
            
            if not combination_quality:
                quality_results['failed_combinations'].append({
                    'detector': detector,
                    'data_group': data_group,
                    'avg_delay': avg_delay,
                    'avg_fpr': avg_fpr,
                    'avg_recall': avg_recall,
                    'meets_delay': meets_delay,
                    'meets_fpr': meets_fpr,
                    'meets_recall': meets_recall
                })
                quality_results['overall_quality'] = False
            
            # Update detector quality tracking
            if detector not in quality_results['detector_quality']:
                quality_results['detector_quality'][detector] = {
                    'total_combinations': 0,
                    'passed_combinations': 0,
                    'avg_delay': [],
                    'avg_fpr': [],
                    'avg_recall': []
                }
            
            detector_quality = quality_results['detector_quality'][detector]
            detector_quality['total_combinations'] += 1
            detector_quality['avg_delay'].append(avg_delay)
            detector_quality['avg_fpr'].append(avg_fpr)
            detector_quality['avg_recall'].append(avg_recall)
            
            if combination_quality:
                detector_quality['passed_combinations'] += 1
        
        # Calculate detector-level statistics
        for detector, stats in quality_results['detector_quality'].items():
            stats['pass_rate'] = stats['passed_combinations'] / stats['total_combinations']
            stats['overall_avg_delay'] = np.mean(stats['avg_delay'])
            stats['overall_avg_fpr'] = np.mean(stats['avg_fpr'])
            stats['overall_avg_recall'] = np.mean(stats['avg_recall'])
        
        # Log quality results
        self.logger.info(f"Overall quality: {'PASS' if quality_results['overall_quality'] else 'FAIL'}")
        self.logger.info(f"Failed combinations: {len(quality_results['failed_combinations'])}")
        
        for detector, stats in quality_results['detector_quality'].items():
            self.logger.info(f"{detector} pass rate: {stats['pass_rate']:.2%}")
        
        return quality_results
    
    def save_all_results(self, raw_results: pd.DataFrame, 
                        aggregated_results: pd.DataFrame,
                        quality_results: Dict[str, Any]):
        """Save all experimental results"""
        
        self.logger.info("Saving experimental results")
        
        # Save raw results
        raw_results_path = os.path.join(
            self.config_manager.paths['raw_results_dir'],
            self.config_manager.output_files['raw_results']
        )
        raw_results.to_csv(raw_results_path, index=False)
        self.logger.info(f"Raw results saved to: {raw_results_path}")
        
        # Save aggregated results
        agg_results_path = os.path.join(
            self.config_manager.paths['aggregated_results_dir'],
            self.config_manager.output_files['aggregated_results']
        )
        aggregated_results.to_csv(agg_results_path, index=False)
        self.logger.info(f"Aggregated results saved to: {agg_results_path}")
        
        # Save quality report
        quality_report_path = os.path.join(
            self.config_manager.paths['results_dir'],
            self.config_manager.output_files['quality_report']
        )
        import json
        with open(quality_report_path, 'w') as f:
            json.dump(quality_results, f, indent=2, default=str)
        self.logger.info(f"Quality report saved to: {quality_report_path}")
        
        # Save summary statistics
        summary_stats = self._generate_summary_statistics(raw_results, aggregated_results)
        summary_stats_path = os.path.join(
            self.config_manager.paths['results_dir'],
            self.config_manager.output_files['summary_stats']
        )
        with open(summary_stats_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        self.logger.info(f"Summary statistics saved to: {summary_stats_path}")
    
    def _generate_summary_statistics(self, raw_results: pd.DataFrame,
                                   aggregated_results: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        successful_results = raw_results[raw_results['success'] == True]
        
        return {
            'experiment_info': {
                'total_experiments': len(raw_results),
                'successful_experiments': len(successful_results),
                'success_rate': len(successful_results) / len(raw_results),
                'execution_time_hours': (time.time() - self.experiment_start_time) / 3600,
                'detectors': list(raw_results['detector'].unique()),
                'data_groups': list(raw_results['data_group'].unique())
            },
            
            'performance_summary': {
                'best_detector_delay': aggregated_results.loc[
                    aggregated_results['drift_detection_delay_mean'].idxmin()
                ][['detector', 'data_group', 'drift_detection_delay_mean']].to_dict(),
                
                'best_detector_fpr': aggregated_results.loc[
                    aggregated_results['false_positive_rate_mean'].idxmin()
                ][['detector', 'data_group', 'false_positive_rate_mean']].to_dict(),
                
                'best_detector_recall': aggregated_results.loc[
                    aggregated_results['drift_detection_recall_mean'].idxmax()
                ][['detector', 'data_group', 'drift_detection_recall_mean']].to_dict()
            },
            
            'detector_rankings': {
                'by_delay': aggregated_results.groupby('detector')['drift_detection_delay_mean'].mean().sort_values().to_dict(),
                'by_fpr': aggregated_results.groupby('detector')['false_positive_rate_mean'].mean().sort_values().to_dict(),
                'by_recall': aggregated_results.groupby('detector')['drift_detection_recall_mean'].mean().sort_values(ascending=False).to_dict()
            },
            
            'imbalance_impact': {
                'delay_by_imbalance': aggregated_results.groupby('imbalance_ratio')['drift_detection_delay_mean'].mean().to_dict(),
                'fpr_by_imbalance': aggregated_results.groupby('imbalance_ratio')['false_positive_rate_mean'].mean().to_dict(),
                'recall_by_imbalance': aggregated_results.groupby('imbalance_ratio')['drift_detection_recall_mean'].mean().to_dict()
            }
        }
    
    def generate_visualizations(self, aggregated_results: pd.DataFrame) -> List[str]:
        """Generate all visualizations"""
        
        self.logger.info("Generating visualizations")
        
        # Create comprehensive report
        figures = self.visualizer.create_experiment_report(
            aggregated_results, 
            self.config_manager.paths['figures_dir']
        )
        
        self.logger.info(f"Generated {len(figures)} visualization files")
        
        return [f"Figure {i+1}" for i in range(len(figures))]
    
    def run_complete_experiment(self) -> Dict[str, Any]:
        """Run complete experimental pipeline"""
        
        self.logger.info("="*60)
        self.logger.info("STARTING COMPLETE EXPERIMENTAL PIPELINE")
        self.logger.info("="*60)
        
        # Validate setup
        if not validate_experiment_setup():
            raise RuntimeError("Experiment setup validation failed")
        
        # Step 1: Run all experiments
        self.logger.info("Step 1: Running all experiments")
        raw_results = self.run_all_experiments()
        
        # Step 2: Aggregate results
        self.logger.info("Step 2: Aggregating results")
        aggregated_results = self.aggregate_results(raw_results)
        
        # Step 3: Evaluate quality
        self.logger.info("Step 3: Evaluating quality")
        quality_results = self.evaluate_quality(aggregated_results)
        
        # Step 4: Save results
        self.logger.info("Step 4: Saving results")
        self.save_all_results(raw_results, aggregated_results, quality_results)
        
        # Step 5: Generate visualizations
        self.logger.info("Step 5: Generating visualizations")
        visualization_files = self.generate_visualizations(aggregated_results)
        
        # Create final summary
        total_time = time.time() - self.experiment_start_time
        final_summary = {
            'experiment_completed': True,
            'total_execution_time_hours': total_time / 3600,
            'total_experiments': len(raw_results),
            'successful_experiments': len(raw_results[raw_results['success'] == True]),
            'overall_quality_pass': quality_results['overall_quality'],
            'failed_combinations': len(quality_results['failed_combinations']),
            'visualization_files': visualization_files,
            'output_files': {
                'raw_results': os.path.join(self.config_manager.paths['raw_results_dir'], 
                                          self.config_manager.output_files['raw_results']),
                'aggregated_results': os.path.join(self.config_manager.paths['aggregated_results_dir'], 
                                                 self.config_manager.output_files['aggregated_results']),
                'quality_report': os.path.join(self.config_manager.paths['results_dir'], 
                                             self.config_manager.output_files['quality_report']),
                'summary_stats': os.path.join(self.config_manager.paths['results_dir'], 
                                            self.config_manager.output_files['summary_stats'])
            }
        }
        
        self.logger.info("="*60)
        self.logger.info("EXPERIMENTAL PIPELINE COMPLETED")
        self.logger.info(f"Total time: {total_time/3600:.2f} hours")
        self.logger.info(f"Overall quality: {'PASS' if quality_results['overall_quality'] else 'FAIL'}")
        self.logger.info("="*60)
        
        return final_summary


def run_experiment_from_config(config_file: str = None) -> Dict[str, Any]:
    """Run experiment from configuration file"""
    
    config_override = {}
    if config_file and os.path.exists(config_file):
        import json
        with open(config_file, 'r') as f:
            config_override = json.load(f)
    
    # Create and run experiment
    runner = ExperimentRunner(config_override)
    return runner.run_complete_experiment()


def run_quick_test() -> Dict[str, Any]:
    """Run quick test with reduced parameters"""
    
    quick_config = {
        'n_runs': 2,  # Reduced for testing
        'data_groups': {
            'Group A1': {
                'attributes': 2,
                'generator': 'SINE1',
                'imbalance_ratio': [5, 5],
                'description': '2D balanced data'
            },
            'Group A2': {
                'attributes': 2,
                'generator': 'SINE1', 
                'imbalance_ratio': [7, 3],
                'description': '2D moderate imbalance'
            }
        },
        'detectors': ['ADWIN', 'DDM'],  # Exclude QuadCDD for quick test
        'stream_length': 1000,  # Shorter streams
        'quality_thresholds': {
            'fpr_threshold': 0.2,  # More lenient
            'delay_threshold': 100
        }
    }
    
    runner = ExperimentRunner(quick_config)
    return runner.run_complete_experiment()


def main():
    """Main function for command-line usage"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Concept Drift Detection Experiments')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced parameters')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate setup without running experiments')
    
    args = parser.parse_args()
    
    if args.validate_only:
        # Just validate setup
        if validate_experiment_setup():
            print("✓ Experiment setup validation passed")
            return 0
        else:
            print("✗ Experiment setup validation failed")
            return 1
    
    if args.quick_test:
        # Run quick test
        print("Running quick test...")
        summary = run_quick_test()
    else:
        # Run full experiment
        print("Running full experiment...")
        summary = run_experiment_from_config(args.config)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Experiment completed: {summary['experiment_completed']}")
    print(f"Total execution time: {summary['total_execution_time_hours']:.2f} hours")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful experiments: {summary['successful_experiments']}")
    print(f"Overall quality: {'PASS' if summary['overall_quality_pass'] else 'FAIL'}")
    print(f"Failed combinations: {summary['failed_combinations']}")
    print("\nOutput files:")
    for file_type, path in summary['output_files'].items():
        print(f"  {file_type}: {path}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())