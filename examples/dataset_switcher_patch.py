import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
import os
import torch

# -- Load dataset based on name --
def load_dataset(name, batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])

    if name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    elif name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    elif name == "SVHN":
        train_dataset = datasets.SVHN(root="data", split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(root="data", split='test', download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # Return datasets along with loaders
    return train_loader, test_loader, train_dataset, test_dataset

# -- Load dataset schedule --
def load_dataset_schedule():
    try:
        with open("dataset_schedule.json", "r") as f:
            return json.load(f)
    except:
        return [{"start_epoch": 0, "dataset": "CIFAR100"}]

# -- Determine current dataset for epoch --
def get_current_dataset(epoch, schedule):
    current = schedule[0]
    for entry in schedule:
        if epoch >= entry["start_epoch"]:
            current = entry
        else:
            break
    return current["dataset"]

# --- Inside your training loop ---
# Replace the static loader line with something like:
# (this should be inside your ARP epoch loop)

# --- Epoch loop with G_ij memory retention ---
dataset_schedule = load_dataset_schedule()
last_dataset = None
start_epoch = 0  # Define the starting epoch
end_epoch = 100  # Define the ending epoch

for epoch in range(start_epoch, end_epoch):
    current_dataset = get_current_dataset(epoch, dataset_schedule)
    if current_dataset != last_dataset:
        train_loader, test_loader, train_dataset, test_dataset = load_dataset(current_dataset)
            # Save G_ij memory for last dataset (if exists)
        if last_dataset is not None:
            g_path = os.path.join("checkpoints", f"g_memory_{last_dataset}.pt")
            torch.save(arp_optimizer.state_dict(), g_path)
            print(f"💾 Saved G_ij memory for {last_dataset}")

        # Load G_ij memory for new dataset (if available)
        g_path = os.path.join("checkpoints", f"g_memory_{current_dataset}.pt")
        if os.path.exists(g_path):
            arp_optimizer.load_state_dict(torch.load(g_path))
            print(f"🔁 Loaded G_ij memory for {current_dataset}")
        else:
            print(f"🆕 No prior G_ij memory found for {current_dataset}, starting fresh")

        print(f"🔄 Switched to dataset: {current_dataset} at epoch {epoch}")
        last_dataset = current_dataset

    # ... then continue training using train_loader/test_loader as usual
