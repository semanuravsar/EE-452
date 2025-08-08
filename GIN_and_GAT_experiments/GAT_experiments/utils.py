from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from seiz_eeg.dataset import EEGDataset

import torch
import random
import os


from torch_geometric.data import Data
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader






################################################################################
# Preprocessing
################################################################################



def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain"""

    bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]

from scipy.signal import stft

def stft_transform(x: np.ndarray, fs=250, nperseg=128, noverlap=64) -> np.ndarray:
    features = []
    for ch in x.T:
        f, t, Zxx = signal.stft(ch, fs=fs, nperseg=nperseg, noverlap=noverlap)
        power = np.abs(Zxx) ** 2
        log_power = np.log1p(np.mean(power, axis=1))  # Mean over time
        features.append(log_power)
    return np.stack(features)  # [channels, freqs]


import pywt

def wavelet_energy(x, wavelet='db4', level=4):
    result = []
    for ch in x.T:
        coeffs = pywt.wavedec(ch, wavelet, level=level)
        energies = [np.log1p(np.sum(np.square(c))) for c in coeffs]
        result.append(energies)
    return np.stack(result)

from scipy.signal import welch

def bandpower(x, fs=250):
    bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]
    result = []
    for ch in x.T:
        f, Pxx = welch(ch, fs=fs, nperseg=256)
        bandpowers = [np.log1p(np.trapezoid(Pxx[(f >= low) & (f < high)], f[(f >= low) & (f < high)])) for low, high in bands]
        result.append(bandpowers)
    return np.stack(result)

"""
def combined_transform(x):
    ""Concatenate wavelet + bandpower + STFT per channel""
    x = signal.sosfiltfilt(bp_filter, x, axis=0)  # time-domain cleaning

    wvlt = wavelet_energy(x)    # shape: [channels, ~5]
    bp = bandpower(x)           # shape: [channels, 4]
    stft = stft_transform(x)    # shape: [channels, freqs]

    # Make sure all have matching shape[0] (channels)
    assert wvlt.shape[0] == bp.shape[0] == stft.shape[0]

    out = np.concatenate([wvlt, bp, stft], axis=1)  # along features
    return out  # final shape: [channels, features_combined]
"""

def normalize_features(feat: np.ndarray, axis=0, eps=1e-8) -> np.ndarray:
    """Z-score normalization per channel or feature"""
    mean = feat.mean(axis=axis, keepdims=True)
    std = feat.std(axis=axis, keepdims=True) + eps
    return (feat - mean) / std

def normalized_combined_transform(x):
    """Concatenate wavelet + bandpower + STFT per channel"""

    bp_filter = signal.butter(4, (0.5, 50), btype="bandpass", output="sos", fs=250)

    x = signal.sosfiltfilt(bp_filter, x, axis=0)  # time-domain cleaning

    # Extract features
    wvlt = wavelet_energy(x)       # [channels, ~5]
    bp   = bandpower(x)            # [channels, 4]
    stft = stft_transform(x)       # [channels, ~60+]

    # Normalize each separately
    # wvlt = normalize_features(wvlt, axis=1)  # normalize across features
    # bp   = normalize_features(bp, axis=1)
    # stft = normalize_features(stft, axis=1)

    # Combine features along feature axis
    out = np.concatenate([wvlt, bp, stft], axis=1)
    out = out[:,:64]

    out[:,30:] *= 250
    out = normalize_features(out, axis=1)
    
    return out


def cropped_stft_transform(x):
    """Crop the STFT transform to only keep the frequencies between 0.5 and 30Hz"""
    out = stft_transform(x)
    out = out[:,33:]
    out = normalize_features(out, axis=1)
    return out



################################################################################
# Data Loading
################################################################################


def seed_everything(seed: int):
    # Python random module
    random.seed(seed)
    # Numpy random module
    np.random.seed(seed)
    # Torch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
class EEGGraphDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_dataset, edge_index, use_supernode=False):
        self.eeg_dataset = eeg_dataset
        self.edge_index = edge_index
        self.use_supernode = use_supernode

    def __len__(self):
        return len(self.eeg_dataset)


    def __getitem__(self, idx):
        x_np, y_or_id = self.eeg_dataset[idx]  # x_np: (19, d), y_or_id: int or str
        x = torch.tensor(x_np, dtype=torch.float)  # EEG features

        if self.use_supernode:
            x_super = x.mean(dim=0, keepdim=True)
            x = torch.cat([x, x_super], dim=0)

            supernode_edges = [(i, 19) for i in range(19)]
            full_edges = self.edge_index + supernode_edges
            edge_index = torch.tensor(full_edges, dtype=torch.long).T
        else:
            edge_index = torch.tensor(self.edge_index, dtype=torch.long).T

        # Return ID (string) instead of label if it's a test set
        if isinstance(y_or_id, str):
            return Data(x=x, edge_index=edge_index), y_or_id
        else:
            y = torch.tensor([y_or_id], dtype=torch.long)
            return Data(x=x, edge_index=edge_index, y=y)
"""  

class EEGGraphDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_dataset, edge_index, use_supernode=False, return_id=False):
        self.eeg_dataset = eeg_dataset
        self.edge_index = edge_index
        self.use_supernode = use_supernode
        self.return_id = return_id

    def __len__(self):
        return len(self.eeg_dataset)

    def __getitem__(self, idx):
        if self.return_id:
            x_np, sample_id = self.eeg_dataset[idx]
        else:
            x_np, y = self.eeg_dataset[idx]

        x = torch.tensor(x_np, dtype=torch.float)

        if self.use_supernode:
            x_super = x.mean(dim=0, keepdim=True)
            x = torch.cat([x, x_super], dim=0)
            supernode_edges = [(i, 19) for i in range(19)]
            full_edges = self.edge_index + supernode_edges
            edge_index = torch.tensor(full_edges, dtype=torch.long).T
        else:
            edge_index = torch.tensor(self.edge_index, dtype=torch.long).T

        data = Data(x=x, edge_index=edge_index)

        if self.return_id:
            return data, sample_id
        else:
            y = torch.tensor([y], dtype=torch.long)
            data.y = y
            return data
