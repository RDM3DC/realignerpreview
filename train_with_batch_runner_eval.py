# --- Dynamic ARP Switch Trainer with Batch Runner + CSV Logger + Fast Test Eval Mode ---

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
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model state from {checkpoint_path}")
    return model

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

# --- Train ARP from checkpoint with dynamic switching and fast test eval ---
def train_arp_with_dynamic_switch(alpha, mu, weight_decay=1e-4, max_epochs=30, spike_threshold=0.1, instability_std=0.2, switch_min_epoch=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideResNet28_10().to(device)
    model = load_model_for_arp_start(model)
    criterion = torch.nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    switched = False

    run_id = f"adamw2arp_alpha{alpha:.2e}_mu{mu:.2e}_lr1e-03_wd1e-02_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

            # Fast test eval mode before ARP switch, then slower eval
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
                loss_improvement = loss_history[-2] - loss_history[-1] if len(loss_history) >= 2 else 0

                # Debugging: Print calculated values for switching conditions
                print(f"Epoch {epoch+1}: delta_acc={delta_acc:.4f}, loss_std={loss_std:.4f}, loss_improvement={loss_improvement:.4f}")

                # Lowered thresholds further
                if delta_acc >= 0.05:  # Reduced from 0.3%
                    print("\n🔥 Spike detected — switching to ARP optimizer\n")
                    optimizer = ARPOptimizer(model.parameters(),
                                             lr=1e-3,
                                             alpha=alpha,
                                             mu=mu,
                                             weight_decay=weight_decay,
                                             clamp_G_min=0.0,
                                             clamp_G_max=10.0)
                    switched = True

                elif loss_std > 0.03:  # Reduced from 0.05
                    print("\n⚠️ Instability detected — aborting run\n")
                    break

                elif loss_improvement < 0.002:  # Reduced from 0.005
                    print("\n⏳ Loss plateau detected — switching to ARP optimizer\n")
                    optimizer = ARPOptimizer(model.parameters(),
                                             lr=1e-3,
                                             alpha=alpha,
                                             mu=mu,
                                             weight_decay=weight_decay,
                                             clamp_G_min=0.0,
                                             clamp_G_max=10.0)
                    switched = True

                # Forced switch at epoch 12 if no switch has occurred
                elif epoch == 12 and not switched:
                    print("\n🔁 Forced switch to ARP optimizer at epoch", epoch + 1, "\n")
                    optimizer = ARPOptimizer(model.parameters(),
                                             lr=1e-3,
                                             alpha=alpha,
                                             mu=mu,
                                             weight_decay=weight_decay,
                                             clamp_G_min=0.0,
                                             clamp_G_max=10.0)
                    switched = True

            # Log smoothness metrics to TensorBoard
            if len(loss_history) >= 5:
                loss_std = np.std(loss_history[-5:])  # Smaller std = smoother
                writer.add_scalar("Smoothness/Loss Std (5)", loss_std, epoch)

            if epoch > 0:
                delta_loss = abs(loss_history[-1] - loss_history[-2])  # Change from previous epoch
                writer.add_scalar("Smoothness/Delta Loss", delta_loss, epoch)

    writer.close()
    print(f"📄 Logged to {csv_log_path}\n")
    return accuracy_history, loss_history

if __name__ == "__main__":
    # Example configuration for testing
    alpha_vals = [0.01, 0.015, 0.02]
    mu_vals = [0.003, 0.005, 0.0068]

    for alpha in alpha_vals:
        for mu in mu_vals:
            print(f"Starting run with alpha={alpha}, mu={mu}")
            train_arp_with_dynamic_switch(alpha=alpha, mu=mu)
