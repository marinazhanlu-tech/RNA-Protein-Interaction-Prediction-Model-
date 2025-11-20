import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Hyperparameters
RNA_MAX_LEN = 200
PROT_MAX_LEN = 500
EMBED_DIM = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Dictionaries for encoding sequences
RNA_ALPHABET = 'AUCG'
PROT_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

# Create encoding dictionaries
rna_to_idx = {char: i+1 for i, char in enumerate(RNA_ALPHABET)}
prot_to_idx = {char: i+1 for i, char in enumerate(PROT_ALPHABET)}

# Function to encode sequence
def encode_seq(seq, seq_dict, max_len):
    seq = seq.upper()
    encoded = np.zeros(max_len, dtype=int)
    for i, char in enumerate(seq):
        if i >= max_len:
            break
        if char in seq_dict:
            encoded[i] = seq_dict[char]
    return encoded

# Custom Dataset
class RNAProteinDataset(Dataset):
    def __init__(self, rna_seqs, prot_seqs, labels):
        self.rna_seqs = [torch.tensor(encode_seq(seq, rna_to_idx, RNA_MAX_LEN), dtype=torch.long) for seq in rna_seqs]
        self.prot_seqs = [torch.tensor(encode_seq(seq, prot_to_idx, PROT_MAX_LEN), dtype=torch.long) for seq in prot_seqs]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.rna_seqs[idx], self.prot_seqs[idx], self.labels[idx]

# Model Definition
class RNAProteinInteractionModel(nn.Module):
    def __init__(self, rna_vocab_size, prot_vocab_size, embed_dim, rna_max_len, prot_max_len):
        super(RNAProteinInteractionModel, self).__init__()
        
        # Embedding layers
        self.rna_embedding = nn.Embedding(rna_vocab_size + 1, embed_dim)
        self.prot_embedding = nn.Embedding(prot_vocab_size + 1, embed_dim)
        
        # CNN layers for feature extraction
        self.rna_cnn = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.prot_cnn = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate the flattened size after CNN and pooling
        rna_output_size = (rna_max_len // 4) * 128  # After two MaxPool1d with kernel_size=2
        prot_output_size = (prot_max_len // 4) * 128
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(rna_output_size + prot_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, rna_seq, prot_seq):
        # Embedding
        rna_emb = self.rna_embedding(rna_seq)
        prot_emb = self.prot_embedding(prot_seq)
        
        # Permute for CNN (batch, channels, length)
        rna_emb = rna_emb.permute(0, 2, 1)
        prot_emb = prot_emb.permute(0, 2, 1)
        
        # CNN feature extraction
        rna_features = self.rna_cnn(rna_emb)
        prot_features = self.prot_cnn(prot_emb)
        
        # Flatten
        rna_flat = rna_features.view(rna_features.size(0), -1)
        prot_flat = prot_features.view(prot_features.size(0), -1)
        
        # Concatenate features
        combined = torch.cat((rna_flat, prot_flat), dim=1)
        
        # Fully connected layers
        output = self.fc(combined)
        
        return output.squeeze()

# Generate dummy data for demonstration
def generate_dummy_data(num_samples=1000):
    rna_seqs = []
    prot_seqs = []
    labels = []
    
    for _ in range(num_samples):
        # Random RNA sequence
        rna_len = np.random.randint(50, RNA_MAX_LEN)
        rna_seq = ''.join(np.random.choice(list(RNA_ALPHABET), rna_len))
        rna_seqs.append(rna_seq)
        
        # Random protein sequence
        prot_len = np.random.randint(50, PROT_MAX_LEN)
        prot_seq = ''.join(np.random.choice(list(PROT_ALPHABET), prot_len))
        prot_seqs.append(prot_seq)
        
        # Random label
        label = np.random.randint(0, 2)
        labels.append(label)
    
    return rna_seqs, prot_seqs, labels

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for rna_seqs, prot_seqs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(rna_seqs, prot_seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for rna_seqs, prot_seqs, labels in val_loader:
                outputs = model(rna_seqs, prot_seqs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        running_loss /= len(train_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    # Generate dummy data
    rna_seqs, prot_seqs, labels = generate_dummy_data(2000)
    
    # Split the data
    X_train_rna, X_val_rna, X_train_prot, X_val_prot, y_train, y_val = train_test_split(
        rna_seqs, prot_seqs, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = RNAProteinDataset(X_train_rna, X_train_prot, y_train)
    val_dataset = RNAProteinDataset(X_val_rna, X_val_prot, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize the model
    model = RNAProteinInteractionModel(
        rna_vocab_size=len(RNA_ALPHABET),
        prot_vocab_size=len(PROT_ALPHABET),
        embed_dim=EMBED_DIM,
        rna_max_len=RNA_MAX_LEN,
        prot_max_len=PROT_MAX_LEN
    )
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

if __name__ == '__main__':
    main()