from typing import Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    X, y = make_circles(
        n_samples=config.n_samples, random_state=config.seed, noise=config.noise
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=config.seed)

    X_train_t = torch.from_numpy(X_train).to(torch.float32)
    y_train_t = torch.from_numpy(y_train).to(torch.float32)
    X_val_t = torch.from_numpy(X_val).to(torch.float32)
    y_val_t = torch.from_numpy(y_val).to(torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)

    return train_dataloader, val_dataloader
