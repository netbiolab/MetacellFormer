"""Paper scDrugMap head, shared by the pooled benchmark and PC9 transfer.
Adapted from the experiment's archived native_head/run.py; architecture and
optimizer loop retained. predict_mlp exposes the original final-epoch logits.
"""
import argparse
import numpy as np
import torch
from torch import nn, optim

def calculate_metrics(y_true, logits):
    return {"truth": y_true, "logits": logits}

class OfficialEBDMLP(nn.Module):
    """Architecture used by official scDrugMap EBD scripts (no dropout)."""

    def __init__(self, input_size: int, hidden_size: int = 512, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float]:
    set_seed(seed)
    model = OfficialEBDMLP(X_train.shape[1], args.hidden_size, 2).to(args.device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Keep the official batch sizes and shuffle semantics, but stage each
    # split once on the selected GPU.  The released script transfers every
    # small batch from CPU to GPU, which makes the hundreds of category-level
    # EBD fits dominated by transfer/launch overhead rather than training.
    X_train_device = torch.from_numpy(X_train).to(args.device)
    labels_train_device = torch.nn.functional.one_hot(
        torch.from_numpy(y_train), num_classes=2
    ).float().to(args.device)
    X_test_device = torch.from_numpy(X_test).to(args.device)
    labels_test_device = torch.from_numpy(y_test).to(args.device)
    shuffle_generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(args.epochs):
        permutation = torch.randperm(len(y_train), generator=shuffle_generator)
        for start in range(0, len(y_train), args.batch_size_train):
            batch_indices = permutation[start : start + args.batch_size_train].to(args.device)
            xb = X_train_device.index_select(0, batch_indices)
            yb = labels_train_device.index_select(0, batch_indices)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(y_test), args.batch_size_test):
            stop = start + args.batch_size_test
            logits.append(model(X_test_device[start:stop]).cpu().numpy())
            labels.append(labels_test_device[start:stop].cpu().numpy())
    result = calculate_metrics(np.concatenate(labels), np.concatenate(logits))
    del model, optimizer, X_train_device, labels_train_device, X_test_device, labels_test_device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
