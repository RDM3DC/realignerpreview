# --- Dynamic ARP Switch Trainer with Batch Runner ---
# Trains with AdamW, freezes at epoch 20, then runs batch ARP optimizer sweeps.

import torch
import numpy as np
import os
from torch.optim import AdamW
from optimizers.arp_optimizer import ARPOptimizer
from models.resnet import WideResNet28_10
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# --- Utility: Save checkpoint at epoch 20 ---
def save_checkpoint(model, optimizer, path='checkpoints/adamw_epoch20.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"✅ Checkpoint saved to {path}")

# --- Utility: Load checkpoint for ARP continuation ---
def load_model_for_arp_start(model, checkpoint_path='checkpoints/adamw_epoch20.pth'):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model state from {checkpoint_path}")
    return model

# --- Train ARP from checkpoint with dynamic switching ---
def train_arp_with_dynamic_switch(alpha, mu, weight_decay=1e-4, max_epochs=30, spike_threshold=1.5, instability_std=0.2, switch_min_epoch=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideResNet28_10().to(device)
    model = load_model_for_arp_start(model)
    criterion = torch.nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

    # Start with AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    switched = False

    run_id = f"adamw2arp_alpha{alpha:.2e}_mu{mu:.2e}_lr1e-03_wd1e-02_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_id}")

    loss_history, accuracy_history = [], []
    for epoch in range(max_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        # Print training progress for each epoch
        if not switched:
            print(f"[AdamW] Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
        else:
            print(f"[ARP α={alpha}, μ={mu}] Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

        # --- Dynamic switch logic ---
        if not switched and epoch >= switch_min_epoch:
            delta_acc = accuracy - accuracy_history[epoch - 1] if epoch > 0 else 0
            loss_std = np.std(loss_history[-3:]) if len(loss_history) >= 3 else 0

            if delta_acc >= spike_threshold:
                print("\n🔥 Spike detected — switching to ARP optimizer\n")
                optimizer = ARPOptimizer(model.parameters(),
                                         lr=1e-3,
                                         alpha=alpha,
                                         mu=mu,
                                         weight_decay=weight_decay,
                                         clamp_G_min=0.0,
                                         clamp_G_max=10.0)
                switched = True

            elif loss_std > instability_std:
                print("\n⚠️ Instability detected — aborting run\n")
                break

    writer.close()
    return accuracy_history, loss_history

# --- Train AdamW to 50 epochs and save checkpoint at epoch 20 ---
def train_adamw_full_and_save_checkpoint(checkpoint_path='checkpoints/adamw_epoch20.pth', max_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideResNet28_10().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # TensorBoard writer
    run_id = f"adamw_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_id}")

    print(f"\n=== Training AdamW for {max_epochs} epochs ===\n")
    for epoch in range(max_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        print(f"[AdamW] Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

        # Save checkpoint at epoch 20
        if epoch + 1 == 20:
            save_checkpoint(model, optimizer, checkpoint_path)

    writer.close()
    return model

# --- Batch Run with Dynamic Switching ---
def batch_run_arp_configs_with_dynamic_switch():
    # Ensure the first AdamW run goes to 50 epochs and saves checkpoint at epoch 20
    checkpoint_path = 'checkpoints/adamw_epoch20.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        train_adamw_full_and_save_checkpoint(checkpoint_path)

    alpha_vals = [0.01, 0.015, 0.02]
    mu_vals = [0.003, 0.005, 0.0068]

    for alpha in alpha_vals:
        for mu in mu_vals:
            train_arp_with_dynamic_switch(alpha=alpha, mu=mu)

if __name__ == "__main__":
    batch_run_arp_configs_with_dynamic_switch()
