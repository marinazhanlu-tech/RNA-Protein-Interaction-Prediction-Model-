import random
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import os
from typing import List, Dict, Any, Union, Tuple, Optional

def is_valid_rna_sequence(sequence: str) -> bool:
    """Check if a string is a valid RNA sequence (contains only A, U, G, C)."""
    return all(base.upper() in {'A', 'U', 'G', 'C'} for base in sequence)

def is_valid_dna_sequence(sequence: str) -> bool:
    """Check if a string is a valid DNA sequence (contains only A, T, G, C)."""
    return all(base.upper() in {'A', 'T', 'G', 'C'} for base in sequence)

def is_valid_protein_sequence(sequence: str) -> bool:
    """Check if a string is a valid protein sequence (contains only standard amino acids)."""
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa in amino_acids for aa in sequence)

def is_valid_fasta_sequence(sequence: str) -> bool:
    """Check if a string is a valid FASTA format sequence."""
    lines = sequence.strip().split('\n')
    if not lines or not lines[0].startswith('>'):
        return False
    return all(is_valid_rna_sequence(line.strip()) for line in lines[1:] if line.strip())

def generate_random_rna_sequence(length: int) -> str:
    """Generate a random RNA sequence of specified length."""
    bases = ['A', 'U', 'G', 'C']
    return ''.join(random.choice(bases) for _ in range(length))

def generate_random_dna_sequence(length: int) -> str:
    """Generate a random DNA sequence of specified length."""
    bases = ['A', 'T', 'G', 'C']
    return ''.join(random.choice(bases) for _ in range(length))

def generate_random_protein_sequence(length: int) -> str:
    """Generate a random protein sequence of specified length."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return ''.join(random.choice(amino_acids) for _ in range(length))

def generate_test_sequences(num_sequences: int, seq_type: str = 'rna') -> List[str]:
    """Generate a list of test sequences of specified type."""
    if seq_type == 'rna':
        return [generate_random_rna_sequence(random.randint(20, 50)) for _ in range(num_sequences)]
    elif seq_type == 'dna':
        return [generate_random_dna_sequence(random.randint(20, 50)) for _ in range(num_sequences)]
    elif seq_type == 'protein':
        return [generate_random_protein_sequence(random.randint(10, 30)) for _ in range(num_sequences)]
    else:
        raise ValueError("Invalid sequence type. Must be 'rna', 'dna', or 'protein'.")

def plot_sequence_length_distribution(sequences: List[str], title: str = "Sequence Length Distribution"):
    """Plot a histogram of sequence lengths."""
    lengths = [len(seq) for seq in sequences]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_nucleotide_composition(sequences: List[str], seq_type: str = 'rna'):
    """Plot the nucleotide composition of sequences."""
    if seq_type not in ['rna', 'dna']:
        raise ValueError("seq_type must be either 'rna' or 'dna'")
    
    bases = ['A', 'U', 'G', 'C'] if seq_type == 'rna' else ['A', 'T', 'G', 'C']
    counts = {base: 0 for base in bases}
    
    for seq in sequences:
        for base in seq.upper():
            if base in counts:
                counts[base] += 1
    
    plt.figure(figsize=(8, 8))
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
    plt.title(f"{seq_type.upper()} Nucleotide Composition")
    plt.show()

def save_sequences_to_fasta(sequences: List[str], filename: str):
    """Save sequences to a FASTA file."""
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences, 1):
            f.write(f">seq_{i}\n")
            f.write(f"{seq}\n")

def load_sequences_from_fasta(filename: str) -> List[str]:
    """Load sequences from a FASTA file."""
    sequences = []
    with open(filename, 'r') as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append(''.join(current_seq))
    return sequences

def save_sequences_to_csv(sequences: List[str], filename: str):
    """Save sequences to a CSV file with one sequence per row."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sequence"])
        for seq in sequences:
            writer.writerow([seq])

def load_sequences_from_csv(filename: str) -> List[str]:
    """Load sequences from a CSV file."""
    sequences = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                sequences.append(row[0])
    return sequences

def save_sequences_to_json(sequences: List[str], filename: str):
    """Save sequences to a JSON file."""
    data = {"sequences": sequences}
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_sequences_from_json(filename: str) -> List[str]:
    """Load sequences from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data.get("sequences", [])

def generate_dummy_data(num_samples: int, rna_alphabet: str = 'AUCG',
                           protein_alphabet: str = 'ARNDCEQGHILKMFPSTWYV',
                           rna_min_len: int = 50, rna_max_len: int = 200,
                           protein_min_len: int = 100, protein_max_len: int = 500,
                           seed: int = 42) -> Tuple[List[str], List[str], List[float]]:
    """
    生成用于测试的虚拟RNA-蛋白质对数据（改进版：更真实、更复杂的数据）

    改进策略：
    1. 正样本：基于统计特征（GC含量、氨基酸组成等），而不是固定模式
    2. 负样本：也有一定的统计特征，但分布不同
    3. 添加噪声和变异，使数据更接近真实情况
    4. 正负样本的差异更小、更模糊

    Args:
        num_samples: 生成的样本数量
        rna_alphabet: RNA字母表
        protein_alphabet: 蛋白质字母表
        rna_min_len: RNA序列最小长度
        rna_max_len: RNA序列最大长度
        protein_min_len: 蛋白质序列最小长度
        protein_max_len: 蛋白质序列最大长度
        seed: 随机种子

    Returns:
        Tuple of (rna_sequences, protein_sequences, labels)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    rna_sequences = []
    protein_sequences = []
    labels = []
    
    num_positives = num_samples // 2
    num_negatives = num_samples - num_positives
    
    # 生成正样本：基于统计特征
    for i in range(num_positives):
        rna_len = random.randint(rna_min_len, rna_max_len)
        protein_len = random.randint(protein_min_len, protein_max_len)
        
        # RNA序列：偏向高GC含量（0.50-0.60），更接近真实，差异更小
        gc_target = random.uniform(0.50, 0.60)
        num_gc = int(rna_len * gc_target)
        num_au = rna_len - num_gc
        
        # 生成GC和AU字符
        gc_chars = ['G', 'C'] * (num_gc // 2) + random.choices(['G', 'C'], k=num_gc % 2)
        au_chars = ['A', 'U'] * (num_au // 2) + random.choices(['A', 'U'], k=num_au % 2)
        rna_chars = gc_chars + au_chars
        random.shuffle(rna_chars)
        rna_seq = ''.join(rna_chars)
        
        # 添加随机变异（模拟真实数据的噪声）
        mutation_rate = 0.05  # 5%的变异率，更接近真实
        rna_list = list(rna_seq)
        for j in range(len(rna_list)):
            if random.random() < mutation_rate:
                rna_list[j] = random.choice(rna_alphabet)
        rna_seq = ''.join(rna_list)
        
        # 蛋白质序列：偏向特定氨基酸组成（疏水性氨基酸比例较高）
        # 定义疏水性氨基酸
        hydrophobic = ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P']
        hydrophilic = ['R', 'N', 'D', 'E', 'Q', 'H', 'K', 'S', 'T', 'Y', 'C', 'G']
        
        # 正样本：疏水性氨基酸比例较高（0.40-0.50），差异更小
        hydrophobic_ratio = random.uniform(0.40, 0.50)
        num_hydrophobic = int(protein_len * hydrophobic_ratio)
        num_hydrophilic = protein_len - num_hydrophobic
        
        protein_chars = (random.choices(hydrophobic, k=num_hydrophobic) + 
                        random.choices(hydrophilic, k=num_hydrophilic))
        random.shuffle(protein_chars)
        protein_seq = ''.join(protein_chars)
        
        # 添加少量随机变异
        protein_list = list(protein_seq)
        for j in range(len(protein_list)):
            if random.random() < mutation_rate:
                protein_list[j] = random.choice(protein_alphabet)
        protein_seq = ''.join(protein_list)
        
        rna_sequences.append(rna_seq)
        protein_sequences.append(protein_seq)
        labels.append(1.0)
    
    # 生成负样本：不同的统计特征
    for i in range(num_negatives):
        rna_len = random.randint(rna_min_len, rna_max_len)
        protein_len = random.randint(protein_min_len, protein_max_len)
        
        # RNA序列：低GC含量（0.40-0.50），与正样本重叠更多，更真实
        gc_target = random.uniform(0.40, 0.50)
        num_gc = int(rna_len * gc_target)
        num_au = rna_len - num_gc
        
        gc_chars = ['G', 'C'] * (num_gc // 2) + random.choices(['G', 'C'], k=num_gc % 2)
        au_chars = ['A', 'U'] * (num_au // 2) + random.choices(['A', 'U'], k=num_au % 2)
        rna_chars = gc_chars + au_chars
        random.shuffle(rna_chars)
        rna_seq = ''.join(rna_chars)
        
        # 添加变异
        mutation_rate = 0.05  # 5%的变异率，更接近真实
        rna_list = list(rna_seq)
        for j in range(len(rna_list)):
            if random.random() < mutation_rate:
                rna_list[j] = random.choice(rna_alphabet)
        rna_seq = ''.join(rna_list)
        
        # 蛋白质序列：疏水性氨基酸比例较低（0.35-0.45），与正样本重叠更多
        hydrophobic = ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P']
        hydrophilic = ['R', 'N', 'D', 'E', 'Q', 'H', 'K', 'S', 'T', 'Y', 'C', 'G']
        
        hydrophobic_ratio = random.uniform(0.35, 0.45)
        num_hydrophobic = int(protein_len * hydrophobic_ratio)
        num_hydrophilic = protein_len - num_hydrophobic
        
        protein_chars = (random.choices(hydrophobic, k=num_hydrophobic) + 
                        random.choices(hydrophilic, k=num_hydrophilic))
        random.shuffle(protein_chars)
        protein_seq = ''.join(protein_chars)
        
        # 添加变异
        protein_list = list(protein_seq)
        for j in range(len(protein_list)):
            if random.random() < mutation_rate:
                protein_list[j] = random.choice(protein_alphabet)
        protein_seq = ''.join(protein_list)
        
        rna_sequences.append(rna_seq)
        protein_sequences.append(protein_seq)
        labels.append(0.0)
    
    # 打乱顺序
    combined = list(zip(rna_sequences, protein_sequences, labels))
    random.shuffle(combined)
    rna_sequences, protein_sequences, labels = zip(*combined)
    
    return list(rna_sequences), list(protein_sequences), list(labels)

def save_data_to_csv(rna_sequences: List[str], protein_sequences: List[str], 
                     labels: List[float], filename: str):
    """Save RNA-protein pairs to CSV file"""
    import pandas as pd
    df = pd.DataFrame({
        'rna_sequence': rna_sequences,
        'protein_sequence': protein_sequences,
        'label': labels
    })
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}: {len(df)} samples")

def load_data_from_csv(filename: str) -> Tuple[List[str], List[str], List[float]]:
    """Load RNA-protein pairs from CSV file"""
    import pandas as pd
    df = pd.read_csv(filename)
    rna_sequences = df['rna_sequence'].tolist()
    protein_sequences = df['protein_sequence'].tolist()
    labels = df['label'].tolist()
    print(f"Data loaded from {filename}: {len(df)} samples")
    return rna_sequences, protein_sequences, labels