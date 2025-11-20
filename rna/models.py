import torch
import torch.nn as nn
import torch.nn.functional as F

class RnaProteinInteractionModel(nn.Module):
    def __init__(self, rna_vocab_size=5, protein_vocab_size=21, rna_embed_dim=64, 
                 protein_embed_dim=64, cnn_channels=128, cnn_kernel_size=5, 
                 dropout_rate=0.2, fusion_dim=256):
        super(RnaProteinInteractionModel, self).__init__()
        
        # RNA embedding and CNN (vocab_size需要包含padding=0，所以是5: 0,1,2,3,4)
        self.rna_embedding = nn.Embedding(rna_vocab_size, rna_embed_dim, padding_idx=0)
        self.rna_cnn = nn.Conv1d(in_channels=rna_embed_dim, 
                                out_channels=cnn_channels, 
                                kernel_size=cnn_kernel_size,
                                padding='same')
        self.rna_bn = nn.BatchNorm1d(cnn_channels)
        self.rna_pool = nn.AdaptiveMaxPool1d(1)
        
        # Protein embedding and CNN (vocab_size需要包含padding=0，所以是21: 0,1-20)
        self.protein_embedding = nn.Embedding(protein_vocab_size, protein_embed_dim, padding_idx=0)
        self.protein_cnn = nn.Conv1d(in_channels=protein_embed_dim, 
                                    out_channels=cnn_channels, 
                                    kernel_size=cnn_kernel_size,
                                    padding='same')
        self.protein_bn = nn.BatchNorm1d(cnn_channels)
        self.protein_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(2 * cnn_channels, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.output = nn.Linear(fusion_dim // 2, 1)
        
    def forward(self, rna_input, protein_input):
        # RNA processing
        rna_embed = self.rna_embedding(rna_input)  # (batch, seq_len, embed_dim)
        rna_embed = rna_embed.transpose(1, 2)      # (batch, embed_dim, seq_len)
        rna_feat = self.rna_cnn(rna_embed)         # (batch, channels, seq_len)
        rna_feat = self.rna_bn(rna_feat)
        rna_feat = F.relu(rna_feat)
        rna_feat = self.rna_pool(rna_feat).squeeze(-1)  # (batch, channels)
        
        # Protein processing
        protein_embed = self.protein_embedding(protein_input)  # (batch, seq_len, embed_dim)
        protein_embed = protein_embed.transpose(1, 2)          # (batch, embed_dim, seq_len)
        protein_feat = self.protein_cnn(protein_embed)         # (batch, channels, seq_len)
        protein_feat = self.protein_bn(protein_feat)
        protein_feat = F.relu(protein_feat)
        protein_feat = self.protein_pool(protein_feat).squeeze(-1)  # (batch, channels)
        
        # Feature fusion
        combined = torch.cat([rna_feat, protein_feat], dim=1)  # (batch, 2*channels)
        fused = self.fusion(combined)
        
        # Output
        output = self.output(fused)  # (batch, 1)
        return torch.sigmoid(output)

if __name__ == "__main__":
    # Test the model
    batch_size = 32
    rna_seq_len = 100
    protein_seq_len = 200
    
    model = RnaProteinInteractionModel()
    rna_input = torch.randint(0, 4, (batch_size, rna_seq_len))
    protein_input = torch.randint(0, 20, (batch_size, protein_seq_len))
    
    output = model(rna_input, protein_input)
    print(output.shape)  # Should be (32, 1)