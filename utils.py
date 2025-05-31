from typing import override
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, input_size: int, num_heads: int):
        super(Attention, self).__init__()
        # Ensure input_size is divisible by num_heads
        if input_size % num_heads != 0:
            input_size = ((input_size + num_heads - 1) // num_heads) * num_heads
            
        self.input_size = input_size
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, x):
        if len(x) > self.input_size:
            x = x[len(x) - self.input_size, -1]
        elif len(x) < self.input_size:
            x = nn.functional.pad(x, (0, self.input_size - len(x)), value = 0)

        # Handle 1D input (sequence of tokens)
        if x.dim() == 1:
            # Reshape to (batch=1, seq_len=1, embed_dim)
            x = x.view(1, -1, 1)  # First reshape to (1, seq_len, 1)
            x = x.expand(-1, -1, self.input_size)  # Expand to proper embedding size
        
        # Pad if necessary
        if x.size(-1) != self.input_size:
            x = nn.functional.pad(x, (0, self.input_size - x.size(-1)))
        
        # MultiheadAttention expects (batch, seq_len, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.squeeze(0)  # Remove batch dimension if it was added

class Encoder(nn.Module):
    def __init__(self, input_size: int, heads: int, layers: int, output_size: int):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.attn = Attention(input_size, heads)
        self.lstm = nn.LSTM(input_size, output_size, layers, batch_first=True)
        self.linear = nn.Linear(output_size, output_size)

    def forward(self, x):
        if len(x) > self.input_size:
            x = x[-self.input_size:]
        elif len(x) < self.input_size:
            x = nn.functional.pad(x, (0, self.input_size - len(x)), value = 0)
        
        # Handle input size mismatch before passing to attention layer
        if x.dim() == 1:
            # If input is 1D, reshape to 3D (batch=1, seq_len=1, features)
            x = x.unsqueeze(0).unsqueeze(0)
        
        x = self.attn(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size: int, layers: int, output_size: int):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, output_size, layers, batch_first=True)
        self.linear = nn.Linear(output_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)

class EncodeDecodeWrapper(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(EncodeDecodeWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
