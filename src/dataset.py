import torch 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, csv_file_path, seq_len, target_ticker="WTI"):
        df = pd.read_csv(csv_file_path, index_col='timestamp')
        if target_ticker not in df.columns:
            raise ValueError(f"Target ticker '{target_ticker}' not found in the dataset columns.")

        self.seq_len = seq_len
        self.target_ticker = target_ticker
        
        # Separate the target column from the feature columns
        self.target_data = df[self.target_ticker].values
        self.feature_data = df.values
        
        print(f"Dataset created. Number of features: {self.feature_data.shape[1]}")
        print(f"Total number of time steps: {len(self.feature_data)}")
    def __len__(self):
        """
        Returns the total number of possible sequences in the dataset.
        """
        # We subtract seq_len because the last possible sequence starts at this index
        return len(self.feature_data) - self.seq_len

    def __getitem__(self, idx):
        """
        Returns a single sample (a sequence and its corresponding target).
        """
        # --- Input Features (x) ---
        # A sequence of data of length seq_len
        # Shape: [seq_len, num_tickers]
        x = self.feature_data[idx : idx + self.seq_len]

        # --- Time Features (t) ---
        # A simple sequence of numbers from 0 to seq_len-1 for Time2Vec
        # Shape: [seq_len, 1]
        t = np.arange(self.seq_len).reshape(-1, 1)
        
        # --- Target Value (y) ---
        # The LogPercentChange of our target ticker for the *next* time step
        # Shape: [1]
        y = self.target_data[idx + self.seq_len]
        
        # --- Convert to PyTorch Tensors ---
        x = torch.tensor(x, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0) # Add a dimension

        return x, t, y