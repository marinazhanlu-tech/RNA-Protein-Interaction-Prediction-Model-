#!/usr/bin/env python3
"""
生成大规模数据集并保存到data文件夹
"""

import os
from utils import generate_dummy_data, save_data_to_csv

def main():
    # 创建data目录
    os.makedirs('data', exist_ok=True)
    
    # 生成50000个样本
    num_samples = 50000
    print(f"Generating {num_samples} samples...")
    
    rna_seqs, protein_seqs, labels = generate_dummy_data(
        num_samples=num_samples,
        rna_alphabet='AUCG',
        protein_alphabet='ARNDCEQGHILKMFPSTWYV',
        rna_min_len=50,
        rna_max_len=200,
        protein_min_len=100,
        protein_max_len=500,
        seed=42
    )
    
    # 保存到CSV文件
    output_file = 'data/rna_protein_data.csv'
    save_data_to_csv(rna_seqs, protein_seqs, labels, output_file)
    
    # 打印统计信息
    print(f"\nData Statistics:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Positive samples: {sum(labels)}")
    print(f"  Negative samples: {len(labels) - sum(labels)}")
    print(f"  Positive ratio: {sum(labels)/len(labels):.2%}")
    print(f"\nData saved to: {output_file}")

if __name__ == '__main__':
    main()

