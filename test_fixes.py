import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import fix_numpy_float


def test_classifier():
    """Test the fixed classifier"""
    print("Testing classifier...")
    
    from models.classifier import OnlineNaiveBayes, generate_accuracy_sequence
    
    # Create simple test data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    print(f"Test data: {n_samples} samples, {np.unique(y, return_counts=True)}")
    
    # Test classifier
    classifier = OnlineNaiveBayes()
    accuracies = []
    
    for i in range(n_samples):
        acc = classifier.partial_fit(X[i], y[i])
        accuracies.append(acc)
    
    print(f"Classifier test passed. Final accuracy: {accuracies[-1]:.3f}")
    
    # Test accuracy sequence generation
    accuracy_seq = generate_accuracy_sequence(X, y)
    print(f"Accuracy sequence generated. Length: {len(accuracy_seq)}, Mean: {np.mean(accuracy_seq):.3f}")
    
    return True

def test_data_generators():
    """Test the fixed data generators"""
    print("\nTesting data generators...")
    
    from data_streams.synthetic_generator import DataStreamFactory
    
    generators_to_test = ['SINE1', 'SEA', 'Circle']
    
    for gen_type in generators_to_test:
        try:
            print(f"Testing {gen_type}...")
            
            generator = DataStreamFactory.create_generator(
                gen_type,
                n_samples=200,
                drift_start=80,
                drift_end=120,
                drift_severity=0.2,
                drift_type='gradual',
                imbalance_ratio=[7, 3],
                random_state=42
            )
            
            df = generator.generate_stream()
            
            # Validate generated data
            assert len(df) == 200, f"Expected 200 samples, got {len(df)}"
            assert 'y' in df.columns, "Missing target column 'y'"
            
            # Check class distribution
            class_counts = df['y'].value_counts()
            print(f"  Class distribution: {dict(class_counts)}")
            
            # Check for both classes
            unique_classes = df['y'].nunique()
            assert unique_classes == 2, f"Expected 2 classes, got {unique_classes}"
            
            # Check for no NaN values
            assert not df.isnull().any().any(), "Generated data contains NaN values"
            
            print(f"  ✓ {gen_type} passed all tests")
            
        except Exception as e:
            print(f"  ✗ {gen_type} failed: {e}")
            return False
    
    return True

def test_pretraining_data_generation():
    """Test pretraining data generation"""
    print("\nTesting pretraining data generation...")
    
    from data_streams.synthetic_generator import DataStreamFactory
    
    try:
        # Generate small amount of training data for testing
        training_data = DataStreamFactory.generate_pretraining_data(
            n_streams_per_type=10,  # Small number for testing
            drift_types=['gradual', 'sudden'],
            drift_severity_range=(0.1, 0.3),
            random_state=42
        )
        
        print(f"Generated {len(training_data)} training streams")
        
        if len(training_data) > 0:
            # Check first training sample
            accuracy_seq, quadruple = training_data[0]
            print(f"First sample - Accuracy sequence length: {len(accuracy_seq)}")
            print(f"First sample - Quadruple: {quadruple}")
            
            # Validate quadruple
            expected_keys = ['Ds', 'De', 'Dv', 'Dt']
            for key in expected_keys:
                assert key in quadruple, f"Missing key {key} in quadruple"
            
            assert 0 <= quadruple['Ds'] <= 1, f"Invalid Ds value: {quadruple['Ds']}"
            assert 0 <= quadruple['De'] <= 1, f"Invalid De value: {quadruple['De']}"
            assert quadruple['Ds'] < quadruple['De'], "Ds should be less than De"
            
            print("  ✓ Pretraining data generation passed all tests")
            return True
        else:
            print("  ✗ No training data generated")
            return False
            
    except Exception as e:
        print(f"  ✗ Pretraining data generation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING FIXED COMPONENTS")
    print("=" * 60)
    
    results = []
    
    # Test classifier
    try:
        results.append(test_classifier())
    except Exception as e:
        print(f"Classifier test failed: {e}")
        results.append(False)
    
    # Test data generators
    try:
        results.append(test_data_generators())
    except Exception as e:
        print(f"Data generator test failed: {e}")
        results.append(False)
    
    # Test pretraining data generation
    try:
        results.append(test_pretraining_data_generation())
    except Exception as e:
        print(f"Pretraining data test failed: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    test_names = ["Classifier", "Data Generators", "Pretraining Data"]
    for name, result in zip(test_names, results):
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    overall_success = all(results)
    print(f"\nOverall: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("\n✓ All tests passed! You can now try running the full experiment.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)