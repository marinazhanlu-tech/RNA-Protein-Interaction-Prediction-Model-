#!/usr/bin/env python3
"""
评估脚本 - 评估训练好的模型
"""

import os
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from models import RnaProteinInteractionModel
from data_loader import RNADataset, create_data_loader
from config import get_config
from utils import generate_dummy_data

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for rna_inputs, protein_inputs, targets in test_loader:
            rna_inputs = rna_inputs.to(device)
            protein_inputs = protein_inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(rna_inputs, protein_inputs)
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(targets.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def main():
    # Configuration
    config = get_config()
    device = torch.device('cpu')
    
    # Load model
    checkpoint_path = os.path.join(config['paths']['model_save_path'], 'model_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = RnaProteinInteractionModel(**config['model']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Load test data from separate test file
    test_data_file = os.path.join('data', 'rna_protein_test.csv')
    if os.path.exists(test_data_file):
        from utils import load_data_from_csv
        print(f'Loading test data from {test_data_file}')
        rna_seqs, protein_seqs, labels = load_data_from_csv(test_data_file)
        print(f'Using {len(labels)} samples for testing (independent test set)')
    else:
        # Fallback: if test file doesn't exist, try to load from main data file
        data_file = os.path.join('data', 'rna_protein_data.csv')
        if os.path.exists(data_file):
            from utils import load_data_from_csv
            from sklearn.model_selection import train_test_split
            print(f'Test file not found, loading from {data_file} and splitting...')
            rna_seqs, protein_seqs, labels = load_data_from_csv(data_file)
            # Use 10% as test set (matching train.py)
            _, rna_seqs, _, protein_seqs, _, labels = train_test_split(
                rna_seqs, protein_seqs, labels, test_size=0.1, random_state=42
            )
            print(f'Using {len(labels)} samples for testing')
        else:
            # Final fallback: generate test data if file doesn't exist
            print(f'Data files not found, generating test data...')
            rna_seqs, protein_seqs, labels = generate_dummy_data(
                num_samples=400,
                rna_alphabet='AUCG',
                protein_alphabet='ARNDCEQGHILKMFPSTWYV',
                seed=999
            )
    
    # Create test dataset
    test_dataset = RNADataset(
        rna_seqs, protein_seqs, labels,
        max_rna_len=config['data']['max_rna_len'],
        max_protein_len=config['data']['max_protein_len']
    )
    test_loader = create_data_loader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Print metrics
    print("="*60)
    print("模型评估结果")
    print("="*60)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print("="*60)
    
    # Create output directory
    os.makedirs(config['paths']['results_path'], exist_ok=True)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.tight_layout()
    plt.savefig(os.path.join(config['paths']['results_path'], 'confusion_matrix.png'))
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['paths']['results_path'], 'roc_curve.png'))
    plt.close()
    
    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='darkorange', lw=2,
             label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['paths']['results_path'], 'pr_curve.png'))
    plt.close()
    
    print(f"\n评估完成！结果已保存到 {config['paths']['results_path']}")

if __name__ == '__main__':
    main()
