# examples/long_run_v2.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from optimizers.arp_optimizer import ARPOptimizer
from torch.utils.tensorboard import SummaryWriter

# --- Config ---
EPOCHS = 1000
BATCH_SIZE = 128
LR = 1e-3
ALPHA = 0.005
MU = 0.002
WEIGHT_DECAY = 1e-2
CHECKPOINT_INTERVAL = 50
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "runs/realignr_longrun_v2"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == '__main__':
    # --- Dataset ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # --- Model ---
    model = torchvision.models.resnet18(num_classes=100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Logger ---
    writer = SummaryWriter(log_dir=LOG_DIR)

    # --- Optimizer ---
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Switch to ARP at epoch 100
        if epoch == 100:
            optimizer = ARPOptimizer(model.parameters(), lr=LR * 0.5, alpha=ALPHA, mu=MU, weight_decay=WEIGHT_DECAY)
            print("🔄 Switched to ARPOptimizer at epoch 100")

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = 100. * correct / total

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        # Print & Log
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

        # --- Checkpoint ---
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"realignr_v2_epoch{epoch+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1
            }, ckpt_path)
            print(f"✅ Checkpoint saved to {ckpt_path}")

    writer.close()