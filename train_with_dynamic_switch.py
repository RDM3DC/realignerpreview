# --- Dynamic ARP Switch Trainer ---
# Trigger ARP takeover when AdamW shows a positive accuracy spike,
# or abort if instability is detected.

def train_with_dynamic_switch(optimizer_type='adamw_arp',
                              model_type='wide_resnet',
                              max_epochs=50,
                              spike_threshold=1.5,  # % gain to trigger ARP
                              instability_std=0.2,   # std threshold to abort
                              switch_min_epoch=5):
    import torch
    import numpy as np
    from torch.optim import AdamW
    from optimizers.arp_optimizer import ARPOptimizer
    from models.wide_resnet import WideResNet28_10
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideResNet28_10().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Data loaders
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    # Initial optimizer: AdamW
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    switched = False
    writer = None
    run_id = None

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

        # TensorBoard logging setup after optimizer identified
        if writer is None:
            run_id = f"adamw2arp_dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            writer = SummaryWriter(log_dir=f"runs/{run_id}")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

        # --- Dynamic switch logic ---
        if not switched and epoch >= switch_min_epoch:
            delta_acc = accuracy - accuracy_history[epoch - 1]
            loss_std = np.std(loss_history[-3:]) if len(loss_history) >= 3 else 0

            if delta_acc >= spike_threshold:
                print("\n🔥 Spike detected — switching to ARP optimizer\n")
                optimizer = ARPOptimizer(model.parameters(),
                                         lr=1e-3,
                                         alpha=0.015,
                                         mu=0.0068,
                                         weight_decay=1e-4,
                                         clamp_G_min=0.0,
                                         clamp_G_max=10.0)
                switched = True

            elif loss_std > instability_std:
                print("\n⚠️ Instability detected — aborting run\n")
                break

    if writer:
        writer.close()
    print("\nTraining completed.")
    return accuracy_history, loss_history, run_id
