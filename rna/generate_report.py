#!/usr/bin/env python3
"""
生成完整的训练和评估报告
"""

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def create_comprehensive_report(results_dir='results', output_file='results/training_report.pdf'):
    """创建PDF格式的综合报告"""
    
    # 检查所有图片文件是否存在
    images = {
        '训练曲线': 'training_curves.png',
        '混淆矩阵': 'confusion_matrix.png',
        'ROC曲线': 'roc_curve.png',
        'PR曲线': 'pr_curve.png',
        '指标总结': 'metrics_summary.png'
    }
    
    existing_images = {}
    for name, filename in images.items():
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            existing_images[name] = path
    
    if not existing_images:
        print("⚠️  未找到任何图片文件")
        return
    
    # 创建PDF报告
    with PdfPages(output_file) as pdf:
        # 第一页：标题页
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'RNA-蛋白质相互作用预测', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.6, '训练和评估报告', 
                ha='center', va='center', fontsize=18)
        fig.text(0.5, 0.4, f'生成时间: {os.popen("date").read().strip()}', 
                ha='center', va='center', fontsize=12)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 添加各个图表
        for name, path in existing_images.items():
            fig = plt.figure(figsize=(11, 8.5))
            img = plt.imread(path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(name, fontsize=16, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"✅ 综合报告已保存到: {output_file}")

if __name__ == '__main__':
    create_comprehensive_report()
