#!/usr/bin/env python3
"""
分析训练和评估结果的合理性
"""

import numpy as np
import matplotlib.pyplot as plt
import re

def analyze_training_log(log_file='logs/training.log'):
    """分析训练日志"""
    epochs = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Train Epoch' in line and 'Average Loss' in line:
                match = re.search(r'Epoch (\d+).*?Average Loss: ([\d.]+)', line)
                if match:
                    epochs.append(int(match.group(1)))
                    train_losses.append(float(match.group(2)))
            
            if 'Validation Epoch' in line and 'Accuracy' in line:
                match = re.search(r'Epoch (\d+).*?Average Loss: ([\d.]+).*?Accuracy: ([\d.]+)', line)
                if match:
                    val_losses.append(float(match.group(2)))
                    val_accuracies.append(float(match.group(3)))
    
    return epochs, train_losses, val_losses, val_accuracies

def check_model_predictions():
    """检查模型预测分布"""
    import torch
    from models import RnaProteinInteractionModel
    from data_loader import RNADataset, create_data_loader
    from utils import generate_dummy_data
    from config import get_config
    
    config = get_config()
    device = torch.device('cpu')
    
    checkpoint = torch.load('models/saved/model_best.pth', map_location=device)
    model = RnaProteinInteractionModel(**config['model']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    rna_seqs, protein_seqs, labels = generate_dummy_data(num_samples=200, rna_alphabet='AUCG', protein_alphabet='ARNDCEQGHILKMFPSTWYV')
    
    test_dataset = RNADataset(rna_seqs, protein_seqs, labels, max_rna_len=config['data']['max_rna_len'], max_protein_len=config['data']['max_protein_len'])
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
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def main():
    print("="*70)
    print("训练和评估结果分析")
    print("="*70)
    
    # 1. 分析训练过程
    print("\n【1. 训练过程分析】")
    epochs, train_losses, val_losses, val_accuracies = analyze_training_log()
    
    if len(epochs) > 0:
        print(f"训练轮数: {len(epochs)} epochs")
        print(f"训练损失: {train_losses[0]:.4f} → {train_losses[-1]:.4f} (下降 {train_losses[0]-train_losses[-1]:.4f})")
        print(f"验证损失: {val_losses[0]:.4f} → {val_losses[-1]:.4f} (变化 {val_losses[-1]-val_losses[0]:.4f})")
        print(f"最佳验证准确率: {max(val_accuracies):.2f}% (Epoch {epochs[val_accuracies.index(max(val_accuracies))]})")
        print(f"最终验证准确率: {val_accuracies[-1]:.2f}%")
        
        # 检查过拟合
        if val_losses[-1] > val_losses[0]:
            print(f"\n⚠️  警告: 验证损失上升 ({val_losses[0]:.4f} → {val_losses[-1]:.4f})")
            print("   这表明模型可能过拟合！")
        
        if val_losses[-1] > train_losses[-1] * 1.5:
            print(f"\n⚠️  警告: 验证损失 ({val_losses[-1]:.4f}) 远大于训练损失 ({train_losses[-1]:.4f})")
            print("   这是明显的过拟合迹象！")
    
    # 2. 分析模型预测
    print("\n【2. 模型预测分析】")
    all_preds, all_labels, all_probs = check_model_predictions()
    
    print(f"测试样本数: {len(all_labels)}")
    print(f"真实标签分布: 正样本={sum(all_labels)}, 负样本={len(all_labels)-sum(all_labels)}")
    print(f"预测分布: 预测为正={sum(all_preds)}, 预测为负={len(all_preds)-sum(all_preds)}")
    print(f"预测概率范围: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    print(f"预测概率平均值: {all_probs.mean():.4f}")
    print(f"预测概率标准差: {all_probs.std():.4f}")
    
    # 检查问题
    print("\n【3. 问题诊断】")
    issues = []
    
    if sum(all_preds) == 0:
        issues.append("❌ 严重问题: 模型预测所有样本为负类（预测为正=0）")
        issues.append("   这导致精确率和召回率都为0")
    
    if all_probs.max() < 0.5:
        issues.append("❌ 严重问题: 所有预测概率都小于0.5")
        issues.append(f"   最大概率: {all_probs.max():.4f} < 0.5")
    
    if all_probs.std() < 0.01:
        issues.append("⚠️  警告: 预测概率方差很小，模型输出几乎相同")
        issues.append(f"   标准差: {all_probs.std():.4f}")
    
    if max(val_accuracies) < 60:
        issues.append("⚠️  警告: 最佳验证准确率低于60%，接近随机猜测水平")
    
    if len(issues) == 0:
        print("✅ 未发现明显问题")
    else:
        for issue in issues:
            print(issue)
    
    # 4. 原因分析
    print("\n【4. 可能的原因】")
    print("1. 数据问题:")
    print("   - 使用的是随机生成的虚拟数据，没有真实的RNA-蛋白质相互作用模式")
    print("   - 标签是随机分配的，模型无法学习到有意义的特征")
    print("   - 数据不平衡可能导致模型偏向预测多数类")
    
    print("\n2. 模型问题:")
    print("   - 模型可能过拟合训练数据")
    print("   - 模型容量可能不适合当前任务")
    print("   - 超参数可能需要调整")
    
    print("\n3. 训练问题:")
    print("   - 训练轮数可能不足或过多")
    print("   - 学习率可能需要调整")
    print("   - 需要更好的正则化策略")
    
    # 5. 建议
    print("\n【5. 改进建议】")
    print("1. 使用真实数据:")
    print("   - 替换虚拟数据为真实的RNA-蛋白质相互作用数据集")
    print("   - 确保数据质量和标签准确性")
    
    print("\n2. 数据增强:")
    print("   - 平衡正负样本比例")
    print("   - 使用数据增强技术")
    
    print("\n3. 模型改进:")
    print("   - 增加dropout率或使用更强的正则化")
    print("   - 尝试不同的模型架构")
    print("   - 调整学习率和训练策略")
    
    print("\n4. 训练策略:")
    print("   - 使用学习率调度器")
    print("   - 增加early stopping的patience")
    print("   - 使用交叉验证")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
