import torch
import torch.nn as nn
import math
from src.patcher import Patcher 

class TransformerBatchNormEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.2, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, src):
        # src shape: [Batch, Seq_Len, D_Model]
        
        # Self Attention Block
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        
        # Apply BatchNorm (Transpose needed: B, L, D -> B, D, L)
        src = src.transpose(1, 2)
        src = self.norm1(src)
        src = src.transpose(1, 2)

        # Feed Forward Block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        
        # Apply BatchNorm
        src = src.transpose(1, 2)
        src = self.norm2(src)
        src = src.transpose(1, 2)
        
        return src

class Transformer_Backbone(nn.Module):
    
    def __init__(self, patch_len, stride, n_channels, seq_len, prediction_len, latent_dimension=128, dropout=0.2):
        super().__init__()
        
        # RevIN
        self.patch_len = patch_len
        self.stride = stride
        self.patcher = Patcher(patch_len=patch_len, stride=stride)
        
        self.n_patches = int((seq_len - patch_len) / stride) + 2
        self.latent_dimension = latent_dimension
        
        # Projection
        self.projection = nn.Linear(in_features=patch_len, out_features=latent_dimension)
        
        # Positional Encoding Apprendibile
        self.W_pos = nn.Parameter(torch.zeros(1, self.n_patches, latent_dimension))
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
        
        # Backbone con BatchNorm
        self.layers = nn.ModuleList([
            TransformerBatchNormEncoderLayer(
                d_model=latent_dimension, 
                nhead=16, 
                dim_feedforward=256, 
                dropout=dropout
            ) for _ in range(3)
        ])

        # 1. Head for Forecasting (Supervisionato) -> Predicts L future points
        self.head_forecast = nn.Linear(in_features=self.n_patches * latent_dimension, out_features=prediction_len)
        
        # 2. Head for Pre-training (Self-Supervised) -> Patch reconstruction (D -> P)
        self.head_pretrain = nn.Linear(in_features=latent_dimension, out_features=patch_len)
    
    def forward(self, x, mask_ratio=0.0):
        # x input shape: [batch, channels, seq_len]

        batch, channels, seq_len = x.shape

        # RevIN
        mean_x = x.mean(dim=2, keepdim=True).detach()
        std_x = x.std(dim=2, keepdim=True).detach()
        x = (x - mean_x) / (std_x + 1e-5)

        # Patching
        x, mask = self.patcher(x, mask_ratio=mask_ratio) # [batch, channels, n_patches, patch_len]

        # Projection 
        x = self.projection(x) # [batch, channels, n_patches, latent_dim]
        
        # Reshape for the Transformer: (Batch * Channels, N_Patches, Latent_Dim)
        n_patches = x.shape[2]
        latent_dim = x.shape[3]
        x = x.view(batch * channels, n_patches, latent_dim)
        
        # Add Positional Embedding
        x = x + self.W_pos

        # Transformer Encoder (con BatchNorm)
        for layer in self.layers:
            x = layer(x)
                
        if mask_ratio > 0:
            
            # x shape: (Batch*Channels, N_Patches, Latent_Dim)
            x = self.head_pretrain(x) # -> (Batch*Channels, N_Patches, Patch_Len)
            
            # Original shapes
            x = x.view(batch, channels, n_patches, self.patch_len)
            
            # Loss requires both patch and mask
            return x, mask

        # No_mask case    
        else:
            # Flatten Head
            x = x.reshape(batch, channels, n_patches * latent_dim)
            x = self.head_forecast(x) # [batch, channels, prediction_len]

            # Denormalize (RevIN inverse) only forecasting
            x = x * std_x + mean_x

            return x

if __name__ == "__main__":
    B, C, L = 32, 7, 336
    model = Transformer_Backbone(patch_len=16, stride=8, n_channels=C, seq_len=L, prediction_len=96)
    x = torch.randn(B, C, L)
    
    # Test 1: Forecasting
    y_pred = model(x, mask_ratio=0.0)
    print(f"Forecast Output: {y_pred.shape} (Expected: {B}, {C}, 96)")
    
    # Test 2: Pre-training
    y_rec, mask = model(x, mask_ratio=0.4)
    print(f"Pretrain Output: {y_rec.shape} (Expected: {B}, {C}, N_Patches, 16)")
    print(f"Mask Shape: {mask.shape}")