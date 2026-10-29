import torch
import torch.nn as nn
import torch.nn.functional as F     
import math

from src.models.unet2d import UnetEncoder, UnetDecoder

# Positional Encoding (sinusoidal)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Multi-head self-attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        qkv = qkv.view(B, T, 3, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)  # (3, B, h, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, h, T, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, h, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, D)    # (B, T, D)
        return self.out_proj(out)    # (B, T, D)

# Feedforward network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

# Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        # self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x, mask=None):
        # x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Transformer(nn.Module):
    """
    Transformer model for predicting a single MUAP from a sequence of EMG windows

    Input shape: (B, W, 2, m, T) 
    Output shape: (B, m, T)

    Output is the contribution from a single source to the final window of EMG
    """
    def __init__(self, cfg):
        super().__init__()

        d_model = cfg.model.d_model
        num_layers = cfg.model.num_layers
        num_heads = cfg.model.num_heads
        d_ff = cfg.model.d_ff

        height = cfg.data.num_channels
        width = cfg.data.window_length

        # Separate encoders for context and target windows
        self.context_encoder = UnetEncoder(in_channels=1, out_channels=512)
        self.target_encoder = UnetEncoder(in_channels=2, out_channels=512)
        # d_model-1 to leave room for the target window encoding
        self.fc1 = nn.Linear(512, d_model-1)
        self.fc2 = nn.Linear(d_model, 512 * (height//8) * (width//8))  # Project back to Unet output shape

        self.transformer_encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
        self.unet_decoder = UnetDecoder(in_channels=512, out_channels=1)

    def forward(self, inputs, mask=None, unet_only=False):
        B, W, C, m, T = inputs.shape  # (B, W, 2, m, T)
        context_windows = inputs[:, :-1, 0, :, :]  # (B, W-1, m, T)
        target_windows = inputs[:, -1, :, :, :]  # (B, 2, m, T)

        # Merge B and W 
        context_windows = context_windows.reshape(B * (W - 1), m, T).unsqueeze(1)  # (B*(W-1), 1, m, T)
        # vis.plot_emg(target_windows[0], title='Target Window')
        x1, x2, x3, target_encodings = self.target_encoder(target_windows, return_hidden=True)  # x4: (B, 512, m//8, T//8)
        if unet_only:
            print("Unet only mode, skipping transformer encoder")
            muap = self.unet_decoder(x1, x2, x3, target_encodings)  # (B, m, T)
            # vis.plot_emg(muap[0], title='Unet Decoder Output')
            # exit()
            return muap.squeeze(1)  # (B, m, T)
        
        context_encodings = self.context_encoder(context_windows, return_hidden=False)  # (B*(W-1), 512, m//8, T//8)

        # Reshape context encodings
        context_encodings = context_encodings.view(B, W - 1, 512, m//8, T//8)  # (B, W-1, 512, m//8, T//8)
        target_encodings = target_encodings.unsqueeze(1)  # (B, 1, 512, m//8, T//8)
        # Concatenate to get full sequence of encodings
        sequence_encodings = torch.cat((context_encodings, target_encodings), dim=1)  # (B, W, 512, m//8, T//8)
        # Global average pooling across m and T dimensions
        sequence_encodings = sequence_encodings.mean(dim=[-1, -2])    # (B, W, 512)
    
        sequence_encodings = self.fc1(sequence_encodings)  # (B, W, d_model-1)
        # Add target window encoding as the last element in the sequence
        target_flag_encoding = torch.zeros((B, W, 1), device=target_windows.device)  # (B, W, 1)
        target_flag_encoding[:, -1, 0] = 1.0  # Set the last element to 1
        sequence_encodings = torch.cat((sequence_encodings, target_flag_encoding), dim=-1) # (B, W, d_model)

        z = self.transformer_encoder(sequence_encodings, mask)  # (B, W, d_model)
        # Get the final encoding for the target window 
        z_final = z[:, -1, :]  # (B, d_model)
        
        # Project back out to shape expected by Unet decoder 
        x4 = self.fc2(z_final)  # (B, 512 * (m//8) * (T//8))
        x4 = x4.view(B, 512, m//8, T//8)

        muap = self.unet_decoder(x1, x2, x3, x4)  # (B, m, T)
        return muap.squeeze(1)


# Example usage
if __name__ == "__main__":
    B, T, D = 32, 100, 512  # Batch size, sequence length, embedding dim
    x = torch.randn(B, T, D)
    encoder = TransformerEncoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048)
    out = encoder(x)  # (B, T, D)
