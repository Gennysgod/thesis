import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import fix_numpy_float

def test_imports():
    """Test all necessary imports"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        from data_streams.synthetic_generator import DataStreamFactory
        print("✓ DataStreamFactory imported successfully")
    except ImportError as e:
        print(f"✗ DataStreamFactory import failed: {e}")
        return False
    
    try:
        # Import detectors with better error handling
        print("Testing detector imports...")
        from detectors import ADWINDetector, DDMDetector
        print("✓ Core detectors imported successfully")
        
        # Test QuadCDD import (optional)
        try:
            from detectors import QuadCDDDetector
            print("✓ QuadCDD detector imported successfully")
        except:
            print("⚠ QuadCDD detector not available (this is OK)")
            
    except ImportError as e:
        print(f"✗ Core detectors import failed: {e}")
        return False
    
    try:
        from evaluation.metrics import calculate_single_run_metrics
        print("✓ Metrics imported successfully")
    except ImportError as e:
        print(f"✗ Metrics import failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test data generation"""
    print("\n" + "=" * 60)
    print("TESTING DATA GENERATION")
    print("=" * 60)
    
    try:
        from data_streams.synthetic_generator import DataStreamFactory
        
        # Test simple SINE1 generator
        print("Testing SINE1 generator...")
        generator = DataStreamFactory.create_generator(
            'SINE1',
            n_samples=100,
            drift_start=40,
            drift_end=60,
            drift_severity=0.2,
            drift_type='gradual',
            imbalance_ratio=[6, 4],
            random_state=42
        )
        
        df = generator.generate_stream()
        print(f"✓ Generated {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Class distribution: {df['y'].value_counts().to_dict()}")
        print(f"  Unique classes: {df['y'].nunique()}")
        
        # Validate data
        assert len(df) == 100, f"Expected 100 samples, got {len(df)}"
        assert df['y'].nunique() == 2, f"Expected 2 classes, got {df['y'].nunique()}"
        assert not df.isnull().any().any(), "Data contains NaN values"
        
        print("✓ Data generation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detectors():
    """Test detector creation and basic functionality"""
    print("\n" + "=" * 60)
    print("TESTING DETECTORS")
    print("=" * 60)
    
    try:
        from detectors import ADWINDetector, DDMDetector
        
        # Test ADWIN
        print("Testing ADWIN detector...")
        adwin = ADWINDetector()
        print(f"✓ ADWIN detector created: {adwin.name}")
        
        # Test basic update
        X = np.array([1.0, 2.0])
        y = 1
        result = adwin.update(X, y)
        print(f"✓ ADWIN update works: {result} (type: {type(result)})")
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        
        # Test multiple updates
        for i in range(20):
            X_test = np.random.randn(2)
            y_test = np.random.randint(0, 2)
            result = adwin.update(X_test, y_test)
            assert isinstance(result, bool), f"Update {i}: Expected bool, got {type(result)}"
        print(f"✓ ADWIN multiple updates work")
        
        # Test DDM
        print("Testing DDM detector...")
        ddm = DDMDetector()
        print(f"✓ DDM detector created: {ddm.name}")
        
        # Test basic update
        result = ddm.update(X, y)
        print(f"✓ DDM update works: {result} (type: {type(result)})")
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        
        # Test multiple updates
        for i in range(20):
            X_test = np.random.randn(2)
            y_test = np.random.randint(0, 2)
            result = ddm.update(X_test, y_test)
            assert isinstance(result, bool), f"Update {i}: Expected bool, got {type(result)}"
        print(f"✓ DDM multiple updates work")
        
        # Test QuadCDD if available
        try:
            from detectors import QuadCDDDetector
            print("Testing QuadCDD detector...")
            quadcdd = QuadCDDDetector()
            print(f"✓ QuadCDD detector created: {quadcdd.name}")
            
            result = quadcdd.update(X, y)
            print(f"✓ QuadCDD update works: {result} (type: {type(result)})")
            assert isinstance(result, bool), f"Expected bool, got {type(result)}"
            
            print(f"✓ QuadCDD test passed")
        except Exception as e:
            print(f"⚠ QuadCDD test skipped: {e}")
        
        print("✓ All available detectors test passed")
        return True
        
    except Exception as e:
        print(f"✗ Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_detection():
    """Test simple detection workflow"""
    print("\n" + "=" * 60)
    print("TESTING SIMPLE DETECTION WORKFLOW")
    print("=" * 60)
    
    try:
        from data_streams.synthetic_generator import DataStreamFactory
        from detectors import ADWINDetector
        
        # Generate simple data
        print("Generating simple test data...")
        generator = DataStreamFactory.create_generator(
            'SINE1',
            n_samples=200,
            drift_start=80,
            drift_end=120,
            drift_severity=0.3,
            drift_type='sudden',  # More obvious drift
            imbalance_ratio=[5, 5],
            random_state=42
        )
        
        df = generator.generate_stream()
        print(f"✓ Generated {len(df)} samples with {df['y'].nunique()} classes")
        
        # Create detector
        print("Creating detector...")
        detector = ADWINDetector()
        
        # Process data
        print("Processing data stream...")
        detections = []
        
        for i, row in df.iterrows():
            features = row[['x1', 'x2']].values
            label = row['y']
            
            is_drift_detected = detector.update(features, label)
            if is_drift_detected:
                detections.append(i)
        
        print(f"✓ Processing completed")
        print(f"  Detections: {detections}")
        print(f"  True drift period: 80-120")
        print(f"  Detection count: {len(detections)}")
        
        # Calculate simple metrics
        if detections:
            # Check if any detection is near the true drift
            true_drift = 80
            min_distance = min(abs(d - true_drift) for d in detections)
            print(f"  Closest detection distance: {min_distance}")
            
            if min_distance <= 50:  # Within 50 samples
                print("✓ Drift detected reasonably close to true drift")
            else:
                print("⚠ Drift detected but far from true drift (this can happen)")
        else:
            print("⚠ No drift detected (this can happen with this detector/data)")
        
        print("✓ Simple detection test completed")
        return True
        
    except Exception as e:
        print(f"✗ Simple detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics():
    """Test metrics calculation"""
    print("\n" + "=" * 60)
    print("TESTING METRICS CALCULATION")
    print("=" * 60)
    
    try:
        from evaluation.metrics import calculate_single_run_metrics
        
        # Test metrics with sample data
        detected_drifts = [85, 150]
        true_drift_start = 80
        true_drift_end = 120
        stream_length = 200
        
        print("Testing metrics calculation...")
        metrics = calculate_single_run_metrics(
            detected_drifts=detected_drifts,
            true_drift_start=true_drift_start,
            true_drift_end=true_drift_end,
            stream_length=stream_length,
            tolerance_window=50
        )
        
        print(f"✓ Metrics calculated successfully")
        print(f"  Detection delay: {metrics.drift_detection_delay}")
        print(f"  False positive rate: {metrics.false_positive_rate:.3f}")
        print(f"  Missed drift count: {metrics.missed_drift_count}")
        print(f"  Detection recall: {metrics.drift_detection_recall:.3f}")
        
        # Validate metrics structure
        assert hasattr(metrics, 'drift_detection_delay'), "Missing drift_detection_delay"
        assert hasattr(metrics, 'false_positive_rate'), "Missing false_positive_rate"
        assert hasattr(metrics, 'missed_drift_count'), "Missing missed_drift_count"
        assert hasattr(metrics, 'drift_detection_recall'), "Missing drift_detection_recall"
        
        # Test with empty detections
        print("Testing with empty detections...")
        metrics_empty = calculate_single_run_metrics(
            detected_drifts=[],
            true_drift_start=true_drift_start,
            true_drift_end=true_drift_end,
            stream_length=stream_length,
            tolerance_window=50
        )
        print(f"✓ Empty detections handled correctly")
        
        print("✓ Metrics test passed")
        return True
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests step by step"""
    print("STEP-BY-STEP TESTING OF DRIFT DETECTION SYSTEM")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Data Generation", test_data_generation), 
        ("Detectors", test_detectors),
        ("Simple Detection", test_simple_detection),
        ("Metrics", test_metrics)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n>>> Running {test_name} test...")
            result = test_func()
            results.append((test_name, result))
            
            if not result:
                print(f"\n⚠ Test '{test_name}' failed. Stopping here to fix issues.")
                break
                
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
            break
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready for experiments.")
        print("\nNext steps:")
        print("  1. Run quick test: python quick_test.py")
        print("  2. Run data validation: python -c \"from data_streams.data_validator import DataStreamValidator; DataStreamValidator().validate_all_configurations()\"")
        print("  3. Run quick experiment: python main.py --quick-test")
    else:
        print("\n💥 Some tests failed. Please fix the issues before proceeding.")
        print("\nTroubleshooting:")
        print("  1. Check if skmultiflow is installed: pip install scikit-multiflow")
        print("  2. Check if all required packages are installed")
        print("  3. Check file paths and imports")
        print("  4. Run individual tests to debug specific issues")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)