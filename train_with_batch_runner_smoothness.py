# --- Dynamic ARP Switch Trainer with Smoothness + Accuracy Variance + Grad Norm + Dataset Switch ---

import torch
import numpy as np
import os
import csv
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
    # Load checkpoint while ignoring mismatched layers
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']

    # Remove incompatible keys for the output layer
    state_dict.pop('linear.weight', None)
    state_dict.pop('linear.bias', None)

    # Load the remaining state_dict
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded model state from {checkpoint_path} (with adjusted output layer)")
    return model

# --- Dataset switcher ---
def get_dataloaders(dataset_name="CIFAR10"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    if dataset_name == "CIFAR100":
        train_set = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
    else:
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    return train_loader, test_loader

# --- Evaluation function ---
def evaluate(model, dataloader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# --- Train ARP from checkpoint with enhanced tracking ---
def train_arp_with_dynamic_switch(alpha, mu, dataset="CIFAR10", weight_decay=1e-4, max_epochs=30, spike_threshold=1.5, instability_std=0.2, switch_min_epoch=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideResNet28_10(num_classes=100).to(device)  # Updated for CIFAR100

    # Check if the checkpoint exists
    checkpoint_path = f"checkpoints/{dataset}_epoch20.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Training from scratch to create it.")
        train_full_and_save_checkpoints(dataset=dataset, max_epochs=20)

    # Load the checkpoint
    model = load_model_for_arp_start(model, checkpoint_path=checkpoint_path)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, test_loader = get_dataloaders(dataset)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    switched = False

    run_id = f"adamw2arp_{dataset}_alpha{alpha:.2e}_mu{mu:.2e}_lr1e-03_wd1e-02_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_id}")

    loss_history, accuracy_history = [], []
    csv_log_path = f"results/{run_id}.csv"
    os.makedirs("results", exist_ok=True)

    with open(csv_log_path, mode='w', newline='') as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow(["Epoch", "Train Loss", "Train Accuracy", "Test Accuracy", "Switched to ARP"])

        for epoch in range(max_epochs):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            grad_norm = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                grad_norm += sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
                optimizer.step()

                total_loss += loss.item()
                correct += outputs.argmax(dim=1).eq(labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            grad_norm = grad_norm / len(train_loader)

            loss_history.append(avg_loss)
            accuracy_history.append(accuracy)

            # Smoothness metrics
            if len(loss_history) >= 5:
                loss_std = np.std(loss_history[-5:])
                writer.add_scalar("Smoothness/Loss Std (5)", loss_std, epoch)
            if epoch > 0:
                delta_loss = abs(loss_history[-1] - loss_history[-2])
                writer.add_scalar("Smoothness/Delta Loss", delta_loss, epoch)
                acc_std = np.std(accuracy_history[-5:]) if len(accuracy_history) >= 5 else 0.0
                writer.add_scalar("Smoothness/Accuracy Std (5)", acc_std, epoch)

            # Gradient tracking
            writer.add_scalar("Gradients/Mean Norm", grad_norm, epoch)

            # Test eval schedule
            test_freq = 1 if not switched else 5
            if epoch % test_freq == 0:
                test_loss, test_acc = evaluate(model, test_loader, device)
                writer.add_scalar("Loss/test", test_loss, epoch)
                writer.add_scalar("Accuracy/test", test_acc, epoch)
            else:
                test_acc = None

            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Accuracy/train", accuracy, epoch)
            writer_csv.writerow([epoch + 1, avg_loss, accuracy, test_acc if test_acc else '', int(switched)])

            if not switched:
                print(f"[AdamW] Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
            else:
                print(f"[ARP α={alpha}, μ={mu}] Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

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
    print(f"📄 Logged to {csv_log_path}\n")
    return accuracy_history, loss_history

# New function to train the model for 100 epochs, save a checkpoint at epoch 20, and save the final checkpoint at the end of training
def train_full_and_save_checkpoints(dataset="CIFAR10", max_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10 if dataset == "CIFAR10" else 100
    model = WideResNet28_10(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, _ = get_dataloaders(dataset)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    run_id = f"adamw_full_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_id}")

    print(f"\n=== Training {dataset} for {max_epochs} epochs ===\n")
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

        # Save the checkpoint ONLY at epoch 20
        if epoch + 1 == 20:
            checkpoint_path = f"checkpoints/{dataset}_epoch20.pth"
            save_checkpoint(model, optimizer, checkpoint_path)
            print(f"✅ Saved checkpoint for {dataset} at epoch 20: {checkpoint_path}")

    # Save final checkpoint (if max_epochs is different from 20)
    if max_epochs != 20:
        checkpoint_path = f"checkpoints/{dataset}_epoch{max_epochs}.pth"
        save_checkpoint(model, optimizer, checkpoint_path)

    writer.close()
    print(f"📄 Training completed for {dataset}. Checkpoints saved.")

if __name__ == "__main__":
    # Example run with CIFAR100 dataset and updated parameters
    train_arp_with_dynamic_switch(
        alpha=0.015,
        mu=0.004,
        dataset="CIFAR100",
        switch_min_epoch=30,
        max_epochs=100
    )
