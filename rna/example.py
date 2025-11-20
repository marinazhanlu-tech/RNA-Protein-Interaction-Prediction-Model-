import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import numpy as np
from typing import Dict, Any, Tuple

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rna.models import RNAModel  # Assuming model definition
from rna.utils import load_config, set_seed  # Assuming utility functions

class RNADataset(Dataset):
    """Simple example dataset for RNA sequences"""
    
    def __init__(self, num_samples: int = 1000, seq_length: int = 100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        # Generate random RNA sequences (A=0, C=1, G=2, U=3)
        self.sequences = torch.randint(0, 4, (num_samples, seq_length))
        # Generate random labels (0 or 1)
        self.labels = torch.randint(0, 2, (num_samples,))
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]

def load_data(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Load and prepare data loaders"""
    train_dataset = RNADataset(num_samples=800)
    val_dataset = RNADataset(num_samples=200)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    
    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(sequences).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * sequences.size(0)
    
    return total_loss / len(dataloader.dataset)

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model and return loss and accuracy"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device).float()
            
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * sequences.size(0)
            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    # 1. Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    set_seed(config["seed"])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Create model
    model = RNAModel(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"]
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # 3. Prepare data
    train_loader, val_loader = load_data(config)
    
    # 4. Train model
    print("Starting training...")
    best_val_loss = float("inf")
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch}/{config['training']['epochs']} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(os.path.dirname(__file__), "best_model.pth"))
    
    # 5. Evaluate model
    print("\nEvaluating best model...")
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_acc = evaluate_model(model, val_loader, device)
    print(f"Final Evaluation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # 6. Make predictions on new data
    print("\nMaking predictions on new data...")
    # Generate 5 random sequences
    test_sequences = torch.randint(0, 4, (5, 100))
    
    model.eval()
    with torch.no_grad():
        test_sequences = test_sequences.to(device)
        predictions = model(test_sequences).squeeze()
        predicted_classes = (predictions > 0.5).float()
    
    print("Sample predictions:")
    for i, (seq, pred) in enumerate(zip(test_sequences, predicted_classes)):
        print(f"Sequence {i+1}: Predicted class = {int(pred.item())} (prob: {torch.sigmoid(pred).item():.4f})")

if __name__ == "__main__":
    main()