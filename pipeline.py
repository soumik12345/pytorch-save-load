import torch.nn as nn
import torch.optim as optim

from rich.console import Console

from utils import seed_everything
from data import create_dataloaders


def setup(config):
    config.seed = 42
    config.n_samples = 1000
    config.noise = 0.02
    config.batch_size = 256
    config.hidden_units = 8
    config.learning_rate = 1e-3
    config.epochs = 50

    console = Console()

    seed_everything(config.seed)

    train_dataloader, val_dataloader = create_dataloaders(config)

    model = nn.Sequential(
        nn.Linear(2, 30), nn.ReLU(), nn.Linear(30, 20), nn.ReLU(), nn.Linear(20, 1)
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    return (
        config,
        console,
        train_dataloader,
        val_dataloader,
        model,
        loss_function,
        optimizer,
    )
