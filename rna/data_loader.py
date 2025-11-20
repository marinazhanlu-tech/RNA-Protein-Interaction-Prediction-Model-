import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import numpy as np

# RNA编码字典（只包含A, U, G, C，其他字符映射为0）
RNA_ALPHABET = {
    'A': 1, 'U': 2, 'G': 3, 'C': 4,
    # 其他字符（包括R, Y, S, W, K, M, B, D, H, V, N, -等）映射为0（padding）
}

# 蛋白质编码字典（只包含20种标准氨基酸，其他映射为0）
PROTEIN_ALPHABET = {
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 
    'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 
    'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 
    'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
    # X和其他字符映射为0（padding）
}

def encode_rna(sequence: str) -> np.ndarray:
    """将RNA序列编码为数值数组（只支持A, U, G, C，其他字符映射为0）"""
    encoded = np.zeros(len(sequence), dtype=np.int64)
    for i, char in enumerate(sequence.upper()):
        encoded[i] = RNA_ALPHABET.get(char, 0)  # 未知字符映射为0
    return encoded

def encode_protein(sequence: str) -> np.ndarray:
    """将蛋白质序列编码为数值数组（只支持20种标准氨基酸，其他映射为0）"""
    encoded = np.zeros(len(sequence), dtype=np.int64)
    for i, char in enumerate(sequence.upper()):
        encoded[i] = PROTEIN_ALPHABET.get(char, 0)  # 未知字符映射为0
    return encoded

def pad_sequences(sequences: List[np.ndarray], max_len: int, padding_value: int = 0) -> np.ndarray:
    """填充序列到相同长度"""
    padded_sequences = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded_sequences[i, :length] = seq[:length] if len(seq) > max_len else seq
    return padded_sequences

class RNADataset(Dataset):
    """RNA-蛋白质交互数据集类"""
    
    def __init__(
        self,
        rna_sequences: List[str],
        protein_sequences: List[str],
        labels: Optional[List[float]] = None,
        max_rna_len: int = 1000,
        max_protein_len: int = 1000
    ):
        self.rna_sequences = rna_sequences
        self.protein_sequences = protein_sequences
        self.labels = labels
        self.max_rna_len = max_rna_len
        self.max_protein_len = max_protein_len
        
        # 预编码所有序列
        self.encoded_rnas = [encode_rna(seq) for seq in rna_sequences]
        self.encoded_proteins = [encode_protein(seq) for seq in protein_sequences]
        
        # 预填充所有序列
        self.padded_rnas = pad_sequences(self.encoded_rnas, max_rna_len)
        self.padded_proteins = pad_sequences(self.encoded_proteins, max_protein_len)

    def __len__(self) -> int:
        return len(self.rna_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        rna = torch.tensor(self.padded_rnas[idx], dtype=torch.long)
        protein = torch.tensor(self.padded_proteins[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float) if self.labels is not None else None
        return rna, protein, label

def collate_fn(batch):
    """Collate function for DataLoader"""
    rnas, proteins, labels = zip(*batch)
    rnas = torch.stack(rnas)
    proteins = torch.stack(proteins)
    labels = torch.stack(labels) if labels[0] is not None else None
    return rnas, proteins, labels

def create_data_loader(
    dataset: RNADataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """创建DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

def preprocess_data(
    rna_sequences: List[str],
    protein_sequences: List[str],
    labels: Optional[List[float]] = None,
    max_rna_len: int = 1000,
    max_protein_len: int = 1000,
    truncate: bool = True
) -> Tuple[List[str], List[str], Optional[List[float]]]:
    """数据预处理函数"""
    processed_rnas = []
    processed_proteins = []
    
    for rna, protein in zip(rna_sequences, protein_sequences):
        # 处理RNA序列
        if truncate and len(rna) > max_rna_len:
            rna = rna[:max_rna_len]
        processed_rnas.append(rna)
        
        # 处理蛋白质序列
        if truncate and len(protein) > max_protein_len:
            protein = protein[:max_protein_len]
        processed_proteins.append(protein)
    
    return processed_rnas, processed_proteins, labels