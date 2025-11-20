#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Define model architecture (must match training)
class RNAProteinModel(nn.Module):
    def __init__(self, rna_vocab_size, protein_vocab_size, embedding_dim=128, hidden_dim=256):
        super(RNAProteinModel, self).__init__()
        
        self.rna_embedding = nn.Embedding(rna_vocab_size, embedding_dim)
        self.protein_embedding = nn.Embedding(protein_vocab_size, embedding_dim)
        
        self.rna_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.protein_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, rna_seq, protein_seq):
        rna_emb = self.rna_embedding(rna_seq)
        protein_emb = self.protein_embedding(protein_seq)
        
        rna_out, _ = self.rna_lstm(rna_emb)
        protein_out, _ = self.protein_lstm(protein_emb)
        
        rna_pooled = torch.mean(rna_out, dim=1)
        protein_pooled = torch.mean(protein_out, dim=1)
        
        combined = torch.cat((rna_pooled, protein_pooled), dim=1)
        
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x).squeeze(1)

class RNADataset(Dataset):
    def __init__(self, rna_seq, protein_seq, rna_tokenizer, protein_tokenizer, max_len=512):
        self.rna_seq = rna_seq
        self.protein_seq = protein_seq
        self.rna_tokenizer = rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.rna_seq)
    
    def __getitem__(self, idx):
        rna = self.rna_seq[idx]
        protein = self.protein_seq[idx]
        
        rna_tokens = self.rna_tokenizer(rna, self.max_len)
        protein_tokens = self.protein_tokenizer(protein, self.max_len)
        
        return {
            'rna': torch.tensor(rna_tokens, dtype=torch.long),
            'protein': torch.tensor(protein_tokens, dtype=torch.long)
        }

def rna_tokenizer(seq, max_len):
    # Simple tokenizer for RNA sequences (A, U, G, C)
    token_map = {'A': 1, 'U': 2, 'G': 3, 'C': 4, 'N': 0}
    tokens = [token_map.get(c, 0) for c in seq.upper()]
    tokens = tokens[:max_len]  # Truncate if too long
    tokens += [0] * (max_len - len(tokens))  # Pad if too short
    return tokens

def protein_tokenizer(seq, max_len):
    # Simple tokenizer for protein sequences (20 amino acids)
    aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
    token_map = {aa: i+1 for i, aa in enumerate(aa_vocab)}
    token_map['X'] = 0  # Unknown
    
    tokens = [token_map.get(c, 0) for c in seq.upper()]
    tokens = tokens[:max_len]  # Truncate if too long
    tokens += [0] * (max_len - len(tokens))  # Pad if too short
    return tokens

def load_model(model_path, rna_vocab_size=5, protein_vocab_size=21):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNAProteinModel(rna_vocab_size, protein_vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict(rna_seq, protein_seq, model, device, max_len=512):
    # Preprocess the sequences
    rna_tokens = rna_tokenizer(rna_seq, max_len)
    protein_tokens = protein_tokenizer(protein_seq, max_len)
    
    # Create dataset and dataloader
    dataset = RNADataset([rna_seq], [protein_seq], rna_tokenizer, protein_tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Make prediction
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for batch in dataloader:
            rna = batch['rna'].to(device)
            protein = batch['protein'].to(device)
            
            outputs = model(rna, protein)
            
            # Convert to numpy and calculate confidence
            probs = outputs.cpu().numpy()
            confidence = np.max(probs, axis=1)
            prediction = (probs > 0.5).astype(int)
            
            predictions.extend(prediction)
            confidences.extend(confidence)
    
    return predictions[0], confidences[0]

def main():
    parser = argparse.ArgumentParser(description='Predict RNA-protein interactions')
    parser.add_argument('--rna', type=str, required=True, help='RNA sequence')
    parser.add_argument('--protein', type=str, required=True, help='Protein sequence')
    parser.add_argument('--model', type=str, default='rna/model.pt', help='Path to trained model')
    parser.add_argument('--output', type=str, default='prediction_result.txt', help='Output file path')
    
    args = parser.parse_args()
    
    # Load the model
    model, device = load_model(args.model)
    
    # Make prediction
    prediction, confidence = predict(args.rna, args.protein, model, device)
    
    # Save results to file
    with open(args.output, 'w') as f:
        f.write(f"RNA sequence: {args.rna}\n")
        f.write(f"Protein sequence: {args.protein}\n")
        f.write(f"Prediction (1=Interaction, 0=No Interaction): {prediction}\n")
        f.write(f"Confidence: {confidence:.4f}\n")
    
    print(f"Prediction completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()