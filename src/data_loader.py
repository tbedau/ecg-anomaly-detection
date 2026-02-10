"""ECG data loading and preprocessing utilities."""

import numpy as np
from torch.utils.data import Dataset
import os


class ECGDataset(Dataset):
    """Dataset for loading ECG recordings from the MIT-BIH database."""

    def __init__(self,data_dir,split="train",transform=None):
        self.data_dir=data_dir
        self.split=split
        self.transform=transform
        self.signals=[]
        self.labels=[]
        self._load_data()

    def _load_data(self):
        split_file=os.path.join(self.data_dir,f"{self.split}_records.npy")
        label_file=os.path.join(self.data_dir,f"{self.split}_labels.npy")
        self.signals=np.load(split_file)
        self.labels=np.load(label_file)

        if self.split=="train":
            noise=np.random.normal(0,0.01,self.signals.shape)
            self.signals=self.signals+noise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        signal=self.signals[idx]
        label=self.labels[idx]

        if self.transform:
            signal=self.transform(signal)

        signal=np.expand_dims(signal,axis=0).astype(np.float32)
        return signal,int(label)


def augment_signal(signal, noise_factor=0.05):
    """Add random noise to an ECG signal for data augmentation."""
    noise = np.random.randn(*signal.shape) * noise_factor
    return signal + noise


def normalize_signal(signal):
    """Normalize ECG signal to zero mean and unit variance."""
    mean=np.mean(signal)
    std=np.std(signal)
    if std==0:
        return signal-mean
    return (signal-mean)/std
