"""CIFAR-10 ResNet-18 benchmark for RealignR.

Tracks:
    1. AdamW
    2. AdaGrad
    3. AdamW→AdaGrad
    4. AdamW→RealignR
    5. AdamW→RealignR + CMA

The script logs epoch metrics to CSV in ``logs/``.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adagrad, AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from optim.realignr import RealignR
from utils.plateau import plateau_detected


def get_dataloaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def run(track: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders()
    model = models.resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    epochs = 200
    scheduler = None
    if track == "adamw":
        opt: optim.Optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        switch_done = True
    elif track == "adagrad":
        opt = Adagrad(model.parameters(), lr=2e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        switch_done = True
    else:
        opt = AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        switch_done = False

    val_hist: List[float] = []
    grad_var_hist: List[float] = []

    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", f"cifar10_resnet18_{track}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "grad_var", "snr"])

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            grad_norm_sum = 0.0
            grad_norm_sq_sum = 0.0
            count = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                out = model(data)
                loss = criterion(out, target)
                loss.backward()

                # Gradient norm statistics
                g_vec = torch.cat([
                    p.grad.view(-1) for p in model.parameters() if p.grad is not None
                ])
                g_norm = g_vec.norm().item()
                grad_norm_sum += g_norm
                grad_norm_sq_sum += g_norm ** 2
                count += 1

                opt.step()
                train_loss += loss.item() * data.size(0)

            if count > 0:
                grad_mean = grad_norm_sum / count
                grad_var = grad_norm_sq_sum / count - grad_mean ** 2
                snr = grad_mean ** 2 / (grad_var + 1e-8)
            else:
                grad_var = 0.0
                snr = 0.0

            train_loss /= len(train_loader.dataset)
            val_loss, val_acc = evaluate(model, val_loader, device)
            val_hist.append(val_loss)
            grad_var_hist.append(grad_var)
            writer.writerow([epoch, train_loss, val_loss, val_acc, grad_var, snr])

            scheduler.step()

            if not switch_done and (
                plateau_detected(val_hist, grad_var_hist) or epoch >= 120
            ):
                lr_final = opt.param_groups[0]["lr"]
                if track == "adamw_to_adagrad":
                    opt = Adagrad(model.parameters(), lr=lr_final)
                elif track == "adamw_to_realignr":
                    opt = RealignR(model.parameters(), lr=lr_final * 0.7, mu=0.1, alpha=1.0)
                elif track == "adamw_to_realignr_cma":
                    opt = RealignR(
                        model.parameters(),
                        lr=lr_final * 0.7,
                        mu=0.1,
                        alpha=1.0,
                        cma_xi=0.1,
                        cma_beta=0.05,
                    )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - epoch)
                switch_done = True

    print(f"Log saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 benchmark")
    parser.add_argument(
        "track",
        type=str,
        default="adamw",
        choices=[
            "adamw",
            "adagrad",
            "adamw_to_adagrad",
            "adamw_to_realignr",
            "adamw_to_realignr_cma",
        ],
        help="Optimization track to run",
    )
    args = parser.parse_args()
    run(args.track)
