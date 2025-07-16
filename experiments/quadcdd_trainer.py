import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from typing import List, Dict, Any, Tuple
import os
import argparse
from tqdm import tqdm
import fix_numpy_float

from data_streams.synthetic_generator import DataStreamFactory
from models.quadcdd_network import QuadCDDDataset, create_quadcdd_model
from models.classifier import generate_accuracy_sequence
from utils.common import ensure_directory, save_model, setup_logging

class QuadCDDPretrainer:
    """Pre-trainer for QuadCDD neural network"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging("quadcdd_pretrainer.log")
        
        # Training data storage
        self.training_data = []
        self.accuracy_sequences = []
        self.quadruple_labels = []
        
        # Model
        self.trainer = None
        
    def generate_pretraining_data(self) -> List[Tuple[np.ndarray, Dict]]:
        """
        Generate pre-training data using synthetic data streams
        Stage 1 of QuadCDD framework
        """
        self.logger.info("Starting pre-training data generation...")
        
        generator_types = self.config.get('generators', ['Circle', 'Sine', 'Hyperplane', 'RandomRBF'])
        drift_types = self.config.get('drift_types', ['sudden', 'gradual', 'early-abrupt'])
        streams_per_type = self.config.get('streams_per_type', 1000)
        drift_severity_range = self.config.get('drift_severity_range', [0.01, 0.3])
        total_streams = self.config.get('total_streams', 4000)
        
        training_data = []
        stream_id = 0
        
        # Calculate streams per combination
        total_combinations = len(generator_types) * len(drift_types)
        streams_per_combination = total_streams // total_combinations
        
        self.logger.info(f"Generating {total_streams} training streams...")
        self.logger.info(f"Generators: {generator_types}")
        self.logger.info(f"Drift types: {drift_types}")
        self.logger.info(f"Streams per combination: {streams_per_combination}")
        
        with tqdm(total=total_streams, desc="Generating training data") as pbar:
            
            for generator_type in generator_types:
                for drift_type in drift_types:
                    
                    for i in range(streams_per_combination):
                        # Random parameters for diversity
                        drift_severity = np.random.uniform(*drift_severity_range)
                        drift_start = np.random.randint(200, 600)  # Earlier start for 1000-sample streams
                        drift_duration = np.random.randint(100, 300)
                        drift_end = min(drift_start + drift_duration, 900)
                        
                        # Ensure minimum gap between start and end
                        if drift_end - drift_start < 50:
                            drift_end = drift_start + 50
                        
                        try:
                            # Create generator
                            generator = DataStreamFactory.create_generator(
                                generator_type,
                                n_samples=1000,
                                drift_start=drift_start,
                                drift_end=drift_end,
                                drift_severity=drift_severity,
                                drift_type=drift_type,
                                imbalance_ratio=[5, 5],  # Balanced for pre-training
                                random_state=42 + stream_id
                            )
                            
                            # Generate stream
                            stream_df = generator.generate_stream()
                            
                            # Extract features and labels
                            feature_cols = [col for col in stream_df.columns if col.startswith('x')]
                            X = stream_df[feature_cols].values
                            y = stream_df['y'].values
                            
                            # Generate accuracy sequence using online classifier
                            accuracy_sequence = generate_accuracy_sequence(X, y)

                            
                            # Create quadruple labels (normalized)
                            Ds = drift_start / 1000.0
                            De = drift_end / 1000.0
                            Dv = drift_severity
                            
                            # Determine Dt based on drift type and duration
                            drift_duration_ratio = (drift_end - drift_start) / 1000.0
                            if drift_type == 'gradual' or drift_duration_ratio > 0.2:
                                Dt = 1.0  # Gradual
                            else:
                                Dt = 0.0  # Sudden/Abrupt
                            
                            quadruple = {
                                'Ds': Ds,
                                'De': De, 
                                'Dv': Dv,
                                'Dt': Dt,
                                'generator_type': generator_type,
                                'drift_type': drift_type,
                                'drift_start': drift_start,
                                'drift_end': drift_end,
                                'stream_id': stream_id
                            }
                            
                            training_data.append((accuracy_sequence, quadruple))
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to generate stream {stream_id}: {e}")
                            continue
                        
                        stream_id += 1
                        pbar.update(1)
                        
                        if stream_id >= total_streams:
                            break
                    
                    if stream_id >= total_streams:
                        break
                
                if stream_id >= total_streams:
                    break
        
        self.logger.info(f"Generated {len(training_data)} training samples")
        self.training_data = training_data
        
        return training_data
    
    def prepare_training_datasets(self, train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation dataloaders"""
        
        if not self.training_data:
            raise ValueError("No training data available. Run generate_pretraining_data() first.")
        
        # Extract sequences and quadruples
        sequences = [data[0] for data in self.training_data]
        quadruples = [data[1] for data in self.training_data]
        
        # Create dataset
        dataset = QuadCDDDataset(sequences, quadruples)
        
        # Split into train and validation
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        batch_size = self.config.get('batch_size', 32)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Use 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train QuadCDD neural network"""
        
        self.logger.info("Starting QuadCDD model training...")
        
        # Create model
        self.trainer = create_quadcdd_model(
            input_size=1,
            hidden_size_1=self.config.get('hidden_size_1', 128),
            hidden_size_2=self.config.get('hidden_size_2', 64),
            output_size=4
        )
        
        # Setup training parameters (matching paper Table 2)
        learning_rate = self.config.get('learning_rate', 1e-3)
        momentum = self.config.get('momentum', 0.9)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        self.trainer.setup_optimizer(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Train model
        epochs = self.config.get('epochs', 50)
        early_stopping_patience = self.config.get('early_stopping_patience', 15)
        
        self.trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience
        )
        
        self.logger.info("Model training completed")
    
    def save_pretrained_model(self, model_path: str) -> None:
        """Save pre-trained model"""
        
        if self.trainer is None:
            raise ValueError("No trained model available")
        
        ensure_directory(os.path.dirname(model_path))
        self.trainer.save_checkpoint(model_path)
        
        # Also save configuration
        config_path = model_path.replace('.pth', '_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Pre-trained model saved to {model_path}")
        self.logger.info(f"Configuration saved to {config_path}")
    
    def evaluate_pretrained_model(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate pre-trained model performance"""
        
        if self.trainer is None:
            raise ValueError("No trained model available")
        
        self.logger.info("Evaluating pre-trained model...")
        
        val_loss = self.trainer.validate(val_loader)
        
        # Additional evaluation metrics
        all_predictions = []
        all_targets = []
        
        self.trainer.model.eval()
        with torch.no_grad():
            for sequences, lengths, targets in val_loader:
                sequences = sequences.to(self.trainer.device)
                lengths = lengths.to(self.trainer.device)
                targets = targets.to(self.trainer.device)
                
                predictions = self.trainer.model(sequences, lengths)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate component-wise errors
        errors = np.abs(predictions - targets)
        
        evaluation_results = {
            'validation_loss': val_loss,
            'ds_mae': np.mean(errors[:, 0]),  # Drift start MAE
            'de_mae': np.mean(errors[:, 1]),  # Drift end MAE  
            'dv_mae': np.mean(errors[:, 2]),  # Drift severity MAE
            'dt_mae': np.mean(errors[:, 3]),  # Drift type MAE
            'overall_mae': np.mean(errors)
        }
        
        self.logger.info("Evaluation Results:")
        for metric, value in evaluation_results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return evaluation_results
    
    def run_complete_pretraining(self, model_save_path: str) -> Dict[str, Any]:
        """Run complete pre-training pipeline"""
        
        self.logger.info("Starting complete QuadCDD pre-training pipeline")
        
        # Step 1: Generate training data
        training_data = self.generate_pretraining_data()
        
        # Step 2: Prepare datasets
        train_loader, val_loader = self.prepare_training_datasets()
        
        # Step 3: Train model
        self.train_model(train_loader, val_loader)
        
        # Step 4: Evaluate model
        evaluation_results = self.evaluate_pretrained_model(val_loader)
        
        # Step 5: Save model
        self.save_pretrained_model(model_save_path)
        
        # Step 6: Create summary
        summary = {
            'training_data_size': len(training_data),
            'model_path': model_save_path,
            'evaluation_results': evaluation_results,
            'config': self.config
        }
        
        self.logger.info("Pre-training pipeline completed successfully")
        
        return summary


def create_pretraining_config() -> Dict[str, Any]:
    """Create default pre-training configuration matching paper Table 2"""
    
    return {
        # Data generation
        'generators': ['Circle', 'Sine', 'Hyperplane', 'RandomRBF'],
        'drift_types': ['sudden', 'gradual', 'early-abrupt'],
        'streams_per_type': 1000,
        'total_streams': 4000,
        'drift_severity_range': [0.01, 0.3],
        'drift_threshold': 0.2,
        
        # Model architecture
        'hidden_size_1': 128,
        'hidden_size_2': 64,
        
        # Training parameters (matching paper Table 2)
        'batch_size': 32,
        'learning_rate': 1e-3,  # Pre-training learning rate
        'epochs': 50,
        'early_stopping_patience': 15,
        'momentum': 0.9,  # SGD momentum
        'weight_decay': 1e-5,
        
        # Fine-tuning parameters (matching paper Table 2)
        'finetune_learning_rate': 1e-2,  # Fine-tuning learning rate
        'finetune_epochs': 20,
        'finetune_batch_size': 16,
        
        # Paths
        'model_save_dir': 'models/pretrained_models/',
        'results_save_dir': 'results/pretraining/'
    }


def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description='QuadCDD Pre-training')
    parser.add_argument('--mode', choices=['pretrain', 'evaluate'], 
                       default='pretrain', help='Mode to run')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, 
                       default='models/pretrained_models/quadcdd_pretrained.pth',
                       help='Path to save/load model')
    parser.add_argument('--streams', type=int, default=4000,
                       help='Total number of training streams')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_pretraining_config()
    
    # Override with command line arguments
    if args.streams:
        config['total_streams'] = args.streams
    
    # Create pre-trainer
    pretrainer = QuadCDDPretrainer(config)
    
    if args.mode == 'pretrain':
        # Run pre-training
        summary = pretrainer.run_complete_pretraining(args.model_path)
        
        # Save summary
        summary_path = args.model_path.replace('.pth', '_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Pre-training completed. Model saved to: {args.model_path}")
        print(f"Summary saved to: {summary_path}")
        
    elif args.mode == 'evaluate':
        # Load and evaluate existing model
        if not os.path.exists(args.model_path):
            print(f"Model not found: {args.model_path}")
            return
        
        # Generate some test data for evaluation
        pretrainer.generate_pretraining_data()
        _, val_loader = pretrainer.prepare_training_datasets()
        
        # Load model
        from models.quadcdd_network import create_quadcdd_model
        trainer = create_quadcdd_model()
        trainer.load_checkpoint(args.model_path)
        pretrainer.trainer = trainer
        
        # Evaluate
        results = pretrainer.evaluate_pretrained_model(val_loader)
        
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()