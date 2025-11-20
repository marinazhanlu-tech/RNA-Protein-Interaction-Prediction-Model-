"""
配置文件 - 包含所有超参数和路径配置
"""

# 模型参数（改进版：增强正则化）
MODEL_CONFIG = {
    'rna_vocab_size': 5,  # 0(padding), 1(A), 2(U), 3(G), 4(C)
    'protein_vocab_size': 21,  # 0(padding), 1-20(20种标准氨基酸)
    'rna_embed_dim': 64,
    'protein_embed_dim': 64,
    'cnn_channels': 128,
    'cnn_kernel_size': 5,
    'dropout_rate': 0.5,  # 增加dropout率防止过拟合
    'fusion_dim': 256
}

# 训练参数（改进版：增强正则化）
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.0005,  # 降低学习率
    'epochs': 50,
    'weight_decay': 1e-4,  # 增加权重衰减
    'early_stopping_patience': 15,  # 增加patience
    'save_best_model': True
}

# 数据参数
DATA_CONFIG = {
    'max_rna_len': 200,
    'max_protein_len': 500,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1
}

# 路径配置
PATH_CONFIG = {
    'data_path': './data',
    'model_save_path': './models/saved',
    'log_path': './logs',
    'results_path': './results'
}

# 设备配置
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0
}

# 完整配置字典（用于方便访问）
CONFIG = {
    'model': MODEL_CONFIG,
    'training': TRAINING_CONFIG,
    'data': DATA_CONFIG,
    'paths': PATH_CONFIG,
    'device': DEVICE_CONFIG
}

def get_config():
    """获取完整配置"""
    return CONFIG

def get_model_config():
    """获取模型配置"""
    return MODEL_CONFIG

def get_training_config():
    """获取训练配置"""
    return TRAINING_CONFIG

def get_data_config():
    """获取数据配置"""
    return DATA_CONFIG