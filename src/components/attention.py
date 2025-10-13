# src/components/attention.py

import torch.nn as nn

class CustomAttention(nn.Module):
    """
    A wrapper for the attention mechanism.
    
    For the full "Stockformer" replication, this should be replaced with
    a ProbSparse Attention implementation from a library or custom code.
    
    For now, we use the standard MultiheadAttention from PyTorch.
    """
    def __init__(self, embed_size, num_heads, dropout):
        super(CustomAttention, self).__init__()
        print("WARNING: Using standard MultiheadAttention. For full replication, replace with a ProbSparse implementation.")
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Expects input shape [batch, seq, features]
        )

    def forward(self, x):
        # In MultiheadAttention, query, key, and value are all the same for self-attention
        attn_output, _ = self.attention(query=x, key=x, value=x)
        return attn_output