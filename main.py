import os
import sys
import argparse
import time
from datetime import datetime
import fix_numpy_float

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_pretrained_model():
    """Check if QuadCDD pre-trained model exists"""
    from experiments.experiment_config import get_config_manager
    config_manager = get_config_manager()
    quadcdd_config = config_manager.detector_configs.get('QuadCDD', {})
    model_path = quadcdd_config.get('model_path', 'models/pretrained_models/quadcdd_pretrained.pth')
    return os.path.exists(model_path), model_path

def run_pretraining():
    """Run QuadCDD pre-training if needed"""
    print("Starting QuadCDD pre-training...")
    from experiments.quadcdd_trainer import QuadCDDPretrainer, create_pretraining_config
    
    # Create pre-training configuration
    config = create_pretraining_config()
    
    # Create pre-trainer
    pretrainer = QuadCDDPretrainer(config)
    
    # Run complete pre-training
    model_path = config['model_save_dir'] + 'quadcdd_pretrained.pth'
    summary = pretrainer.run_complete_pretraining(model_path)
    
    print(f"Pre-training completed. Model saved to: {model_path}")
    print(f"Training data size: {summary['training_data_size']}")
    print(f"Validation loss: {summary['evaluation_results']['validation_loss']:.4f}")
    
    return model_path

def run_data_validation():
    """Run comprehensive data validation"""
    print("\n" + "="*70)
    print("RUNNING DATA VALIDATION")
    print("="*70)
    
    from data_streams.data_validator import DataStreamValidator
    
    validator = DataStreamValidator()
    
    # Validate all configurations
    summary_df = validator.validate_all_configurations()
    
    print("\n✓ Data validation completed")
    print("All validation plots saved to: results/validation/")
    
    # Check if all validations passed
    all_passed = True
    for _, row in summary_df.iterrows():
        if row['ratio_difference'] > 0.05:
            print(f"⚠️ Warning: {row['group_name']} has high ratio difference: {row['ratio_difference']:.3f}")
            all_passed = False
        # 移除了错误的 drift_detected 检查，因为该字段在数据验证中不存在
    
    if all_passed:
        print("\n✓ All data validations passed")
    else:
        print("\n⚠️ Some data validations have warnings, but continuing...")
    
    return all_passed

def run_experiment(config_file=None, quick_test=False):
    """Run the main experiment"""
    print("\nStarting main experiment...")
    from experiments.run_experiment import ExperimentRunner, run_quick_test
    
    if quick_test:
        summary = run_quick_test()
    else:
        config_override = {}
        if config_file and os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config_override = json.load(f)
        
        runner = ExperimentRunner(config_override)
        summary = runner.run_complete_experiment()
    
    return summary

def setup_experiment_environment():
    """Setup experiment environment and check dependencies"""
    print("Setting up experiment environment...")
    
    # Check required directories
    required_dirs = [
        'data_streams',
        'detectors', 
        'models',
        'evaluation',
        'experiments',
        'results',
        'utils'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    
    # Check Python packages
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'sklearn', 'river', 'torch', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"✗ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ Environment setup complete")
    return True

def print_experiment_info():
    """Print experiment information"""
    print("="*70)
    print("ADAPTIVE FINANCIAL FRAUD DETECTION EXPERIMENT")
    print("QuadCDD-based Concept Drift Detection with Class Imbalance")
    print("="*70)
    print("This experiment evaluates concept drift detection methods on")
    print("imbalanced data streams, comparing ADWIN, DDM, and QuadCDD detectors.")
    print()
    print("Experiment Components:")
    print("- Data Generation: SINE1 (2D) and SEA (3D) generators")
    print("- Class Imbalance: 5:5, 7:3, 9:1 ratios")
    print("- Detectors: ADWIN, DDM, QuadCDD")
    print("- Metrics: Detection Delay, FPR, Missed Count, Recall")
    print("- Runs: 20 per detector-data combination")
    print("="*70)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Concept Drift Detection Experiment Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline
  python main.py --quick-test       # Run quick test
  python main.py --pretrain-only    # Only run pre-training
  python main.py --validate-only    # Only validate data
  python main.py --skip-validation  # Skip data validation
  python main.py --config config.json  # Use custom configuration
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced parameters')
    parser.add_argument('--pretrain-only', action='store_true',
                       help='Only run QuadCDD pre-training')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run data validation')
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='Skip pre-training (use existing model)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip data validation')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Print experiment info
    if not args.quick_test:
        print_experiment_info()
    
    start_time = time.time()
    
    try:
        # Step 0: Setup environment
        if not setup_experiment_environment():
            return 1
        
        # Step 1: Data validation
        if args.validate_only:
            run_data_validation()
            print("\nData validation completed. Exiting.")
            return 0
        
        if not args.skip_validation and not args.pretrain_only:
            run_data_validation()
        
        # Step 2: Pre-training check and execution
        if not args.skip_pretrain:
            model_exists, model_path = check_pretrained_model()
            if not model_exists:
                print(f"\nQuadCDD pre-trained model not found at: {model_path}")
                print("Running pre-training...")
                model_path = run_pretraining()
            else:
                print(f"\n✓ QuadCDD pre-trained model found at: {model_path}")
        
        if args.pretrain_only:
            print("\nPre-training completed. Exiting.")
            return 0
        
        # Step 3: Run main experiment
        print(f"\nStarting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary = run_experiment(args.config, args.quick_test)
        
        # Step 4: Print final results
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Total execution time: {total_time/60:.2f} minutes")
        print(f"Experiments run: {summary['successful_experiments']}/{summary['total_experiments']}")
        print(f"Overall quality: {'PASS' if summary['overall_quality_pass'] else 'FAIL'}")
        
        if summary['failed_combinations'] > 0:
            print(f"⚠️  {summary['failed_combinations']} combinations failed quality thresholds")
        
        print("\n📊 Results saved to:")
        for file_type, path in summary['output_files'].items():
            if os.path.exists(path):
                print(f"   {file_type}: {path}")
        
        print(f"\n📈 Visualizations: {len(summary['visualization_files'])} files in results/figures/")
        
        print("\n💡 Next steps:")
        print("   1. Review results in results/aggregated_results/")
        print("   2. Check visualizations in results/figures/")
        print("   3. Analyze quality report for failed combinations")
        print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())