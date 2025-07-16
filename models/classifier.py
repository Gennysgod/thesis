import numpy as np
from river import tree
from typing import List

class OnlineHoeffdingTree:
    """Online Hoeffding Tree classifier for accuracy sequence generation"""

    def __init__(self):
        self.classifier = tree.HoeffdingTreeClassifier()
        self.is_fitted = False
        self.accuracy_history = []
        self.seen_classes = set()
        self.sample_count = 0
        self.min_samples_for_prediction = 2

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Incrementally fit the classifier and return current accuracy

        Args:
            X: Feature vector (can be 1D or 2D)
            y: Target value (single value or array)

        Returns:
            Current accuracy on recent data
        """
        # Ensure X is 1D for river (it expects dict-like input)
        if X.ndim > 1:
            X = X.flatten()
        
        # Convert to dict format for river
        if len(X) == 1:
            x_dict = {'x0': float(X[0])}
        else:
            x_dict = {f'x{i}': float(X[i]) for i in range(len(X))}
        
        # Ensure y is scalar
        if np.isarray(y):
            y_val = y.item() if y.size == 1 else y[0]
        else:
            y_val = y
        
        y_val = int(y_val)
        
        self.sample_count += 1
        self.seen_classes.add(y_val)
        
        # Make prediction before updating (if we have enough samples)
        accuracy = 0.5  # Default accuracy
        
        if self.sample_count > self.min_samples_for_prediction:
            try:
                prediction = self.classifier.predict_one(x_dict)
                if prediction is not None:
                    accuracy = 1.0 if prediction == y_val else 0.0
                else:
                    accuracy = 0.5
            except Exception:
                accuracy = 0.5
        
        # Update the model
        try:
            self.classifier.learn_one(x_dict, y_val)
            self.is_fitted = True
        except Exception as e:
            print(f"Warning: Classifier update error: {e}")
        
        self.accuracy_history.append(accuracy)
        return accuracy

    def get_accuracy_sequence(self) -> np.ndarray:
        """Get accuracy sequence over time"""
        return np.array(self.accuracy_history)

    def reset(self):
        """Reset classifier state"""
        self.classifier = tree.HoeffdingTreeClassifier()
        self.is_fitted = False
        self.accuracy_history = []
        self.seen_classes = set()
        self.sample_count = 0

# Keep the old name for compatibility
OnlineNaiveBayes = OnlineHoeffdingTree

def generate_accuracy_sequence(data_stream: np.ndarray,
                             labels: np.ndarray) -> np.ndarray:
    """
    Generate accuracy sequence for a data stream using online classifier

    Args:
        data_stream: Feature matrix (n_samples, n_features)
        labels: Target labels (n_samples,)

    Returns:
        Accuracy sequence over time
    """
    classifier = OnlineHoeffdingTree()
    accuracy_sequence = []

    # Validate input data
    if len(data_stream) != len(labels):
        raise ValueError("Data stream and labels must have the same length")
    
    if len(data_stream) == 0:
        return np.array([])

    # Check if we have valid labels
    unique_labels = np.unique(labels)
    if len(unique_labels) == 0:
        return np.array([0.5] * len(labels))

    for i in range(len(data_stream)):
        X_i = data_stream[i]
        y_i = labels[i]
        
        # Skip invalid data points
        if np.any(np.isnan(X_i)) or np.any(np.isinf(X_i)) or np.isnan(y_i) or np.isinf(y_i):
            accuracy_sequence.append(0.5)  # Default accuracy for invalid data
            continue
            
        try:
            accuracy = classifier.partial_fit(X_i, y_i)
            accuracy_sequence.append(accuracy)
        except Exception as e:
            print(f"Warning: Error at sample {i}: {e}")
            accuracy_sequence.append(0.5)  # Default accuracy on error

    return np.array(accuracy_sequence)