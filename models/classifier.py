import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from typing import List

class OnlineNaiveBayes:
    """Online Naive Bayes classifier for accuracy sequence generation"""

    def __init__(self):
        self.classifier = GaussianNB()
        self.is_fitted = False
        self.accuracy_history = []
        self.seen_classes = set()
        self.prediction_buffer = []
        self.label_buffer = []
        self.min_samples_per_class = 2  # Minimum samples needed per class before prediction

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Incrementally fit the classifier and return current accuracy

        Args:
            X: Feature vector (can be 1D or 2D)
            y: Target value (single value or array)

        Returns:
            Current accuracy on recent data
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Ensure y is array
        if np.isscalar(y):
            y = np.array([y])

        # Add to buffer
        self.prediction_buffer.extend(X)
        self.label_buffer.extend(y)
        
        # Update seen classes
        self.seen_classes.update(y)

        # Check if we have enough samples for both classes
        unique_labels, counts = np.unique(self.label_buffer, return_counts=True)
        
        # Initialize accuracy
        accuracy = 0.5  # Default accuracy when we can't make predictions

        if len(unique_labels) >= 2 and all(count >= self.min_samples_per_class for count in counts):
            # We have enough samples from multiple classes to train and predict
            try:
                # Convert buffer to arrays
                X_buffer = np.array(self.prediction_buffer)
                y_buffer = np.array(self.label_buffer)

                if not self.is_fitted:
                    # First fit with all available classes
                    all_classes = np.array(sorted(list(self.seen_classes)))
                    self.classifier.fit(X_buffer, y_buffer)
                    self.is_fitted = True
                else:
                    # Update with new data
                    # Use partial_fit with all known classes
                    all_classes = np.array(sorted(list(self.seen_classes)))
                    self.classifier.partial_fit(X_buffer[-len(X):], y, classes=all_classes)

                # Make prediction on current sample
                if len(X_buffer) > 0:
                    predictions = self.classifier.predict(X)
                    accuracy = np.mean(predictions == y)
                
            except Exception as e:
                print(f"Warning: Classifier error: {e}")
                # Use a simple baseline accuracy
                accuracy = np.mean(y == np.round(np.mean(self.label_buffer)))
        
        elif len(unique_labels) == 1:
            # Only one class seen so far - perfect accuracy if predicting that class
            accuracy = 1.0 if np.all(y == unique_labels[0]) else 0.0
        
        # Limit buffer size to prevent memory issues
        max_buffer_size = 1000
        if len(self.prediction_buffer) > max_buffer_size:
            # Keep only recent samples
            keep_size = max_buffer_size // 2
            self.prediction_buffer = self.prediction_buffer[-keep_size:]
            self.label_buffer = self.label_buffer[-keep_size:]

        self.accuracy_history.append(accuracy)
        return accuracy

    def get_accuracy_sequence(self) -> np.ndarray:
        """Get accuracy sequence over time"""
        return np.array(self.accuracy_history)

    def reset(self):
        """Reset classifier state"""
        self.classifier = GaussianNB()
        self.is_fitted = False
        self.accuracy_history = []
        self.seen_classes = set()
        self.prediction_buffer = []
        self.label_buffer = []


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
    classifier = OnlineNaiveBayes()
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