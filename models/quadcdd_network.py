import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any
import os

class QuadCDDDataset(Dataset):
    """Dataset for QuadCDD training"""

    def __init__(self, accuracy_sequences: List[np.ndarray],
                 quadruples: List[Dict[str, float]]):
        self.sequences = accuracy_sequences
        self.quadruples = quadruples

        # Find max sequence length for padding
        self.max_length = max(len(seq) for seq in accuracy_sequences)

        # Pad sequences and convert to tensors
        self.padded_sequences = []
        self.sequence_lengths = []

        for seq in accuracy_sequences:
            padded = np.zeros(self.max_length)
            padded[:len(seq)] = seq
            self.padded_sequences.append(padded)
            self.sequence_lengths.append(len(seq))

        self.padded_sequences = np.array(self.padded_sequences)
        self.sequence_lengths = np.array(self.sequence_lengths)

        # Convert quadruples to arrays
        self.targets = np.array([
            [quad['Ds'], quad['De'], quad['Dv'], quad['Dt']]
            for quad in quadruples
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.padded_sequences[idx]).unsqueeze(-1),  # (seq_len, 1)
            torch.LongTensor([self.sequence_lengths[idx]]),  # Keep as 1D tensor
            torch.FloatTensor(self.targets[idx])
        )


class QuadCDDNetwork(nn.Module):
    """BiLSTM network for QuadCDD quadruple prediction"""

    def __init__(self, input_size: int = 1, hidden_size_1: int = 128,
                 hidden_size_2: int = 64, output_size: int = 4,
                 num_layers: int = 1, dropout: float = 0.2):
        super(QuadCDDNetwork, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers

        # First BiLSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Second BiLSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1 * 2,  # bidirectional output
            hidden_size=hidden_size_2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Linear(hidden_size_2 * 2, output_size)  # bidirectional output

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Activation functions
        self.sigmoid = nn.Sigmoid()  # For Ds, De, Dv (0-1 range)

    def forward(self, x, lengths=None):
        """
        Forward pass
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            lengths: Actual sequence lengths (for packed sequences)
        """
        batch_size = x.size(0)

        # Fix lengths tensor dimension issue
        if lengths is not None:
            # Ensure lengths is 1D and on CPU
            lengths = lengths.squeeze()
            if lengths.dim() == 0:  # If scalar, make it 1D
                lengths = lengths.unsqueeze(0)
            lengths = lengths.cpu()

            # Only pack if we have valid lengths
            if len(lengths) > 0 and torch.all(lengths > 0):
                try:
                    x = nn.utils.rnn.pack_padded_sequence(
                        x, lengths, batch_first=True, enforce_sorted=False
                    )
                    use_packed = True
                except Exception as e:
                    print(f"Warning: Failed to pack sequences: {e}")
                    use_packed = False
            else:
                use_packed = False
        else:
            use_packed = False

        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)

        # Unpack if packed
        if use_packed and isinstance(lstm1_out, nn.utils.rnn.PackedSequence):
            lstm1_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm1_out, batch_first=True
            )

        # Apply dropout
        lstm1_out = self.dropout(lstm1_out)

        # Pack again for second layer if we were using packing
        if use_packed and lengths is not None:
            try:
                lstm1_out = nn.utils.rnn.pack_padded_sequence(
                    lstm1_out, lengths, batch_first=True, enforce_sorted=False
                )
            except Exception:
                use_packed = False

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)

        # Unpack if packed
        if use_packed and isinstance(lstm2_out, nn.utils.rnn.PackedSequence):
            lstm2_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm2_out, batch_first=True
            )

        # Take the last output (or mean of all outputs)
        if lengths is not None and not use_packed:
            # Use actual last output for each sequence
            last_outputs = []
            for i, length in enumerate(lengths):
                if length > 0:
                    last_outputs.append(lstm2_out[i, min(length-1, lstm2_out.size(1)-1), :])
                else:
                    last_outputs.append(lstm2_out[i, -1, :])
            last_output = torch.stack(last_outputs)
        else:
            # Use last timestep
            last_output = lstm2_out[:, -1, :]

        # Apply dropout
        last_output = self.dropout(last_output)

        # Final linear layer
        output = self.fc(last_output)

        # Apply activations
        # Ds, De, Dv should be in [0, 1] range
        ds_de_dv = self.sigmoid(output[:, :3])
        # Dt is binary (0 or 1)
        dt = torch.sigmoid(output[:, 3:4])

        return torch.cat([ds_de_dv, dt], dim=1)


class QuadCDDLoss(nn.Module):
    """Custom loss function for QuadCDD with weighted components"""

    def __init__(self, alpha: List[float] = [0.3, 0.3, 0.25, 0.15]):
        super(QuadCDDLoss, self).__init__()
        self.alpha = torch.tensor(alpha)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Calculate weighted square root of sum of squared differences
        Args:
            predictions: Model predictions (batch_size, 4)
            targets: Target quadruples (batch_size, 4)
        """
        # Move alpha to same device as predictions
        if self.alpha.device != predictions.device:
            self.alpha = self.alpha.to(predictions.device)

        # Calculate squared differences
        squared_diffs = (predictions - targets) ** 2

        # Apply weights
        weighted_squared_diffs = self.alpha * squared_diffs

        # Sum and take square root
        loss = torch.sqrt(torch.sum(weighted_squared_diffs, dim=1))

        # Return mean loss across batch
        return torch.mean(loss)


class QuadCDDTrainer:
    """Training and inference wrapper for QuadCDD network"""

    def __init__(self, model: QuadCDDNetwork, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = QuadCDDLoss()
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []

    def setup_optimizer(self, learning_rate: float = 1e-3,
                       weight_decay: float = 1e-5):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, (sequences, lengths, targets) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            lengths = lengths.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sequences, lengths)
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # Print progress
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, dataloader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for sequences, lengths, targets in dataloader:
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(sequences, lengths)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None,
              epochs: int = 50, early_stopping_patience: int = 15):
        """Full training loop"""
        print(f"Training QuadCDD model on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)

            # Train
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)

            # Validate
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)

                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint('best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping after {epoch+1} epochs")
                        break
            else:
                print(f"Train Loss: {train_loss:.4f}")

    def predict(self, accuracy_sequence: np.ndarray) -> Dict[str, float]:
        """Predict quadruple for a single accuracy sequence"""
        self.model.eval()

        # Prepare input - ensure proper tensor dimensions
        sequence_tensor = torch.FloatTensor(accuracy_sequence).unsqueeze(0).unsqueeze(-1)
        length_tensor = torch.LongTensor([len(accuracy_sequence)])  # Keep as 1D

        sequence_tensor = sequence_tensor.to(self.device)
        length_tensor = length_tensor.to(self.device)

        with torch.no_grad():
            try:
                prediction = self.model(sequence_tensor, length_tensor)
                prediction = prediction.cpu().numpy()[0]

                return {
                    'Ds': float(prediction[0]),
                    'De': float(prediction[1]),
                    'Dv': float(prediction[2]),
                    'Dt': float(prediction[3])
                }
            except Exception as e:
                print(f"Error in prediction: {e}")
                # Return default values
                return {
                    'Ds': 0.5,
                    'De': 0.7,
                    'Dv': 0.1,
                    'Dt': 1.0
                }

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'input_size': self.model.lstm1.input_size,
                'hidden_size_1': self.model.hidden_size_1,
                'hidden_size_2': self.model.hidden_size_2,
                'output_size': self.model.fc.out_features,
                'num_layers': self.model.num_layers
            }
        }

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        print(f"Model loaded from {filepath}")

    def fine_tune(self, train_dataloader: DataLoader,
                  learning_rate: float = 1e-2, epochs: int = 10):
        """Fine-tune pre-trained model on new data"""
        print("Fine-tuning QuadCDD model...")

        # Setup optimizer with higher learning rate for fine-tuning
        self.setup_optimizer(learning_rate=learning_rate)

        for epoch in range(epochs):
            print(f"\nFine-tuning Epoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_dataloader)
            print(f"Train Loss: {train_loss:.4f}")

        print("Fine-tuning completed")


def create_quadcdd_model(input_size: int = 1, hidden_size_1: int = 128,
                        hidden_size_2: int = 64, output_size: int = 4) -> QuadCDDTrainer:
    """Factory function to create QuadCDD model and trainer"""
    model = QuadCDDNetwork(
        input_size=input_size,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        output_size=output_size
    )

    trainer = QuadCDDTrainer(model)
    trainer.setup_optimizer()

    return trainer