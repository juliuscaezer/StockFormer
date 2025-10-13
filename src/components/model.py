# src/components/model.py

import torch.nn as nn

# Import the components from their new, separate files
from .time2vec import Time2Vec
from .attention import CustomAttention

class Stockformer(nn.Module):
    """
    The main Stockformer model architecture, now importing its components.
    """
    def __init__(self, num_stocks, seq_len, embed_size, num_heads, num_encoder_layers, dropout):
        super(Stockformer, self).__init__()

        # 1. 1D-CNN for local feature extraction
        self.cnn_feature_extractor = nn.Conv1d(
            in_channels=num_stocks,
            out_channels=embed_size,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.cnn_activation = nn.ReLU()

        # 2. Time2Vec for temporal encoding (imported)
        self.time2vec = Time2Vec(in_features=1, out_features=embed_size)
        
        # 3. Transformer Encoder
        # We create a standard encoder layer but will use our custom attention inside if needed.
        # For simplicity with the default PyTorch TransformerEncoder, we'll let it manage its own standard attention.
        # If you were to build the encoder from scratch, you would insert your CustomAttention module here.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Output Layer
        self.output_layer = nn.Linear(embed_size * seq_len, 1)

    def forward(self, x, t):
        # x shape: [batch_size, seq_len, num_stocks]
        # t shape: [batch_size, seq_len, 1]

        # CNN Path
        x_permuted = x.permute(0, 2, 1)
        cnn_out = self.cnn_feature_extractor(x_permuted)
        cnn_out = self.cnn_activation(cnn_out)
        cnn_out = cnn_out.permute(0, 2, 1)

        # Time2Vec Path
        time_embedding = self.time2vec(t)

        # Combine features and temporal info
        combined_features = cnn_out + time_embedding

        # Transformer Encoder
        encoder_out = self.transformer_encoder(combined_features)

        # Flatten and pass to output layer
        flat_out = encoder_out.reshape(encoder_out.size(0), -1)
        prediction = self.output_layer(flat_out)

        return prediction