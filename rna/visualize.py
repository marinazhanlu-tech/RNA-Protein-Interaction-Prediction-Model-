#!/usr/bin/env python3
"""
训练和测试结果可视化脚本
绘制训练损失、验证损失、准确率等图表
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_training_log(log_file):
    """从训练日志中解析数据（只提取最后一次完整的训练）"""
    # 读取所有行
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 找到最后一次训练完成的位置
    last_completed_idx = None
    for i in range(len(lines)-1, -1, -1):
        if 'Training completed!' in lines[i]:
            last_completed_idx = i
            break
    
    if last_completed_idx is None:
        return [], [], [], []
    
    # 往前找这次训练的开始
    training_start_idx = None
    search_start = min(last_completed_idx, len(lines)-1)
    for i in range(search_start, -1, -1):
        if 'Training on device:' in lines[i]:
            training_start_idx = i
            break
    
    if training_start_idx is None:
        return [], [], [], []
    
    # 提取这次训练的所有epoch数据（使用字典避免重复和乱序）
    train_data = {}
    val_data = {}
    
    for i in range(training_start_idx, last_completed_idx+1):
        # 解析训练损失
        if 'Train Epoch' in lines[i] and 'Average Loss' in lines[i]:
            match = re.search(r'Train Epoch (\d+):.*?Average Loss: ([\d.]+)', lines[i])
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                train_data[epoch] = loss
        
        # 解析验证损失和准确率
        if 'Validation Epoch' in lines[i] and 'Accuracy' in lines[i]:
            match = re.search(r'Validation Epoch (\d+):.*?Average Loss: ([\d.]+).*?Accuracy: ([\d.]+)', lines[i])
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))
                val_data[epoch] = (loss, acc)
    
    # 按epoch顺序排序
    all_epochs = sorted(set(list(train_data.keys()) + list(val_data.keys())))
    
    epochs = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in all_epochs:
        if epoch in train_data and epoch in val_data:
            epochs.append(epoch)
            train_losses.append(train_data[epoch])
            val_losses.append(val_data[epoch][0])
            val_accuracies.append(val_data[epoch][1])
    
    return epochs, train_losses, val_losses, val_accuracies

def plot_training_curves(epochs, train_losses, val_losses, val_accuracies, save_dir='results'):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    axes[0].plot(epochs[:len(val_losses)], val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(epochs[:len(val_accuracies)], val_accuracies, 'g-', label='Validation Accuracy', linewidth=2, marker='^', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存到: {os.path.join(save_dir, 'training_curves.png')}")
    plt.close()

def evaluate_model_and_plot(model_path, data_loader, device, save_dir='results'):
    """评估模型并绘制评估指标"""
    from models import RnaProteinInteractionModel
    from config import get_config
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    config = get_config()
    checkpoint = torch.load(model_path, map_location=device)
    model = RnaProteinInteractionModel(**config['model']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 收集预测结果
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for rna_inputs, protein_inputs, labels in data_loader:
            rna_inputs = rna_inputs.to(device)
            protein_inputs = protein_inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(rna_inputs, protein_inputs)
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Interaction', 'Interaction'],
                yticklabels=['No Interaction', 'Interaction'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵已保存到: {os.path.join(save_dir, 'confusion_matrix.png')}")
    plt.close()
    
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    print(f"✅ ROC曲线已保存到: {os.path.join(save_dir, 'roc_curve.png')}")
    plt.close()
    
    # 绘制PR曲线
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    print(f"✅ PR曲线已保存到: {os.path.join(save_dir, 'pr_curve.png')}")
    plt.close()
    
    # 计算并打印评估指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n" + "="*50)
    print("模型评估指标")
    print("="*50)
    print(f"准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数:            {f1:.4f}")
    print(f"ROC AUC:           {roc_auc:.4f}")
    print(f"PR AUC:            {pr_auc:.4f}")
    print("="*50)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def plot_metrics_summary(metrics, save_dir='results'):
    """绘制指标总结图"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数', 'ROC AUC', 'PR AUC']
    metrics_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['roc_auc'],
        metrics['pr_auc']
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_names)))
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('模型评估指标总结', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 指标总结图已保存到: {os.path.join(save_dir, 'metrics_summary.png')}")
    plt.close()

def main():
    """主函数"""
    import argparse
    from data_loader import RNADataset, create_data_loader
    from utils import generate_dummy_data
    from config import get_config
    
    parser = argparse.ArgumentParser(description='可视化训练和测试结果')
    parser.add_argument('--log_file', type=str, default='logs/training.log', help='训练日志文件路径')
    parser.add_argument('--model_path', type=str, default='models/saved/model_best.pth', help='模型文件路径')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--eval', action='store_true', help='是否进行模型评估')
    
    args = parser.parse_args()
    
    print("="*60)
    print("训练和测试结果可视化")
    print("="*60)
    
    # 1. 绘制训练曲线
    if os.path.exists(args.log_file):
        print(f"\n📊 正在解析训练日志: {args.log_file}")
        epochs, train_losses, val_losses, val_accuracies = parse_training_log(args.log_file)
        
        if epochs:
            print(f"   找到 {len(epochs)} 个epoch的训练数据")
            plot_training_curves(epochs, train_losses, val_losses, val_accuracies, args.save_dir)
        else:
            print("   ⚠️  未找到训练数据")
    else:
        print(f"   ⚠️  训练日志文件不存在: {args.log_file}")
    
    # 2. 评估模型并绘制评估指标
    if args.eval and os.path.exists(args.model_path):
        print(f"\n📈 正在评估模型: {args.model_path}")
        
        config = get_config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 准备测试数据
        rna_seqs, protein_seqs, labels = generate_dummy_data(
            num_samples=200,
            rna_alphabet='AUCG',
            protein_alphabet='ARNDCEQGHILKMFPSTWYV'
        )
        
        test_dataset = RNADataset(
            rna_seqs, protein_seqs, labels,
            max_rna_len=config['data']['max_rna_len'],
            max_protein_len=config['data']['max_protein_len']
        )
        
        test_loader = create_data_loader(test_dataset, batch_size=32, shuffle=False)
        
        # 评估并绘制
        metrics = evaluate_model_and_plot(args.model_path, test_loader, device, args.save_dir)
        plot_metrics_summary(metrics, args.save_dir)
    
    print(f"\n✅ 所有图表已保存到: {args.save_dir}/")
    print("="*60)

if __name__ == '__main__':
    main()
