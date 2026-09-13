import numpy as np
import torch

def create_sliding_windows(hr_norm: np.ndarray, spo2_norm: np.ndarray, seq_len: int = 60, step: int = 10):
    """
    Slices continuous 1D streams into 2D windows: [batch_size, seq_len, 2].
    """
    stacked = np.stack([hr_norm, spo2_norm], axis=1)  # Shape: [N, 2]
    windows = []
    
    for start_idx in range(0, len(stacked) - seq_len + 1, step):
        chunk = stacked[start_idx : start_idx + seq_len]
        windows.append(chunk)
        
    if not windows:
        return torch.empty((0, seq_len, 2), dtype=torch.float32)
        
    return torch.tensor(np.array(windows), dtype=torch.float32)