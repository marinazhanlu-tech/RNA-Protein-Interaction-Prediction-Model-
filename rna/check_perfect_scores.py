#!/usr/bin/env python3
"""
检查为什么指标都是1.0 - 分析是否正常
"""

import torch
import numpy as np
from models import RnaProteinInteractionModel
from data_loader import RNADataset, create_data_loader
from utils import generate_dummy_data
from config import get_config
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def analyze_perfect_scores():
    """分析完美指标的原因"""
    
    config = get_config()
    device = torch.device('cpu')
    
    # 加载模型
    checkpoint = torch.load('models/saved/model_best.pth', map_location=device)
    model = RnaProteinInteractionModel(**config['model']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 使用不同的测试数据（不同的随机种子）
    print("="*70)
    print("分析完美指标的原因")
    print("="*70)
    
    results = []
    
    for seed in [42, 123, 456, 789, 999]:
        rna_seqs, protein_seqs, labels = generate_dummy_data(
            num_samples=200, 
            seed=seed  # 不同的随机种子
        )
        
        test_dataset = RNADataset(
            rna_seqs, protein_seqs, labels,
            max_rna_len=config['data']['max_rna_len'],
            max_protein_len=config['data']['max_protein_len']
        )
        test_loader = create_data_loader(test_dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for rna_inputs, protein_inputs, labels_batch in test_loader:
                rna_inputs = rna_inputs.to(device)
                protein_inputs = protein_inputs.to(device)
                outputs = model(rna_inputs, protein_inputs)
                probs = outputs.squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_labels.extend(labels_batch.numpy())
                all_probs.extend(probs)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall_curve, precision_curve)
        
        results.append({
            'seed': seed,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'prob_range': (all_probs.min(), all_probs.max()),
            'prob_mean': all_probs.mean()
        })
        
        print(f"\n随机种子 {seed}:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  PR AUC: {pr_auc:.4f}")
        print(f"  预测概率范围: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    
    print("\n" + "="*70)
    print("分析结论")
    print("="*70)
    
    avg_roc_auc = np.mean([r['roc_auc'] for r in results])
    avg_pr_auc = np.mean([r['pr_auc'] for r in results])
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    
    print(f"平均准确率: {avg_accuracy:.4f}")
    print(f"平均ROC AUC: {avg_roc_auc:.4f}")
    print(f"平均PR AUC: {avg_pr_auc:.4f}")
    
    if avg_roc_auc > 0.99:
        print("\n⚠️  警告: 指标接近完美可能的原因:")
        print("1. 数据模式过于简单:")
        print("   - 正样本包含明显的模式（特定k-mer）")
        print("   - 负样本完全随机")
        print("   - 模式差异太大，模型很容易区分")
        print()
        print("2. 数据生成方式:")
        print("   - 虚拟数据是按照固定规则生成的")
        print("   - 模式是人工设计的，不是真实的生物模式")
        print("   - 模型学习的是人工规则，不是真实特征")
        print()
        print("3. 这是正常的（对于虚拟数据）:")
        print("   - 如果数据模式明显，完美指标是可能的")
        print("   - 但这不代表模型在真实数据上也会这么好")
        print("   - 真实数据会更复杂，模式更模糊")
    
    return results

if __name__ == '__main__':
    analyze_perfect_scores()
