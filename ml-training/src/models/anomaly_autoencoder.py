import torch
import torch.nn as nn

class HrSpo2Autoencoder(nn.Module):
    def __init__(self, seq_len=60, feat_dim=2, hidden=16):
        super().__init__()
        # Encoder: compresses the 60-second HR/SpO2 window into a hidden state
        self.encoder = nn.LSTM(feat_dim, hidden, batch_first=True)
        # Decoder: attempts to reconstruct the original sequence from the hidden state
        self.decoder = nn.LSTM(hidden, feat_dim, batch_first=True)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, features]
        # h shape: [num_layers, batch_size, hidden_size]
        _, (h, c) = self.encoder(x)
        
        # We take the final hidden state and repeat it across the sequence length
        h_rep = h.permute(1, 0, 2).repeat(1, x.size(1), 1)
        
        # The decoder tries to recreate the input sequence
        out, _ = self.decoder(h_rep)
        return out