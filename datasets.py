import torch
from torch import Tensor
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        waveform = self.df['waveform'].iloc[idx]
        waveform = Tensor(waveform)
        label = self.df['label'].iloc[idx]

        return waveform, label


class MagnitudeEstimationDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        waveform = self.df['waveform'].iloc[idx]
        waveform = Tensor(waveform)

        magnitude = self.df['source_magnitude'].iloc[idx]
        return waveform, magnitude
