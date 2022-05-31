import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function,
    optimizer,
    batch_size,
):
    model.train()
    predictions, losses, accuracy_scores = np.array([]), [], []
    for idx, (X_batch, y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_function(output.flatten(), y_batch)
        loss.backward()
        losses.append(loss.detach().flatten()[0])
        optimizer.step()
        probs = torch.sigmoid(output)
        preds = (probs > 0.5).type(torch.long)
        predictions = np.hstack((predictions, preds.numpy().flatten()))
        accuracy = accuracy_score(y_batch, preds.flatten())
        accuracy_scores.append(accuracy)
    return model, optimizer, losses, accuracy_scores


def val_step(model: nn.Module, dataloader: DataLoader, loss_function, batch_size):
    model.eval()
    predictions, losses, accuracy_scores = np.array([]), [], []
    for idx, (X_batch, y_batch) in enumerate(dataloader):
        output = model(X_batch)
        losses.append(
            F.binary_cross_entropy_with_logits(output.flatten(), y_batch)
            .detach()
            .numpy()
        )
        probs = torch.sigmoid(output)
        preds = (probs > 0.5).type(torch.long)
        predictions = np.hstack((predictions, preds.numpy().flatten()))
        accuracy_scores.append(accuracy_score(y_batch, preds.flatten()))
    return losses, accuracy_scores


def train_and_validate(
    model: nn.Module,
    batch_size: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_function,
    optimizer,
    max_epochs,
    epoch_progress,
    epoch_task,
):
    loss_history, accuracy_history = [], []
    val_loss_history, val_accuracy_history = [], []
    epoch_progress.start_task(0)
    for epoch in range(max_epochs):
        # Train Step
        model, optimizer, losses, accuracy_scores = train_step(
            model,
            train_dataloader,
            loss_function,
            optimizer,
            batch_size,
        )
        wandb.log({"Train Loss": np.mean(losses)})
        wandb.log({"Train Accuracy": np.mean(accuracy_scores)})
        loss_history += losses
        accuracy_history += accuracy_scores
        # Validation Step
        losses, accuracy_scores = val_step(
            model, val_dataloader, loss_function, batch_size
        )
        wandb.log({"Validation Loss": np.mean(losses)})
        wandb.log({"Validation Accuracy": np.mean(accuracy_scores)})
        val_loss_history += losses
        val_accuracy_history += accuracy_scores
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "checkpoint.pth",
        )
        artifact = wandb.Artifact(
            "checkpoints", type="model", metadata={"epoch": epoch}
        )
        artifact.add_file("checkpoint.pth")
        wandb.log_artifact(artifact)
        epoch_progress.update(epoch_task, advance=1)
    return (
        model,
        loss_history,
        accuracy_history,
        val_loss_history,
        val_accuracy_history,
        epoch_progress,
        epoch_task,
    )
