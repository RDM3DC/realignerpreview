# arp_resume_CPRv4_staged.py
import torch
import numpy as np
import os
import csv
from torch.optim import AdamW
from optimizers.arp_optimizer import ARPOptimizer
from models.resnet import WideResNet28_10
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import tempfile
import shutil

# --- DDP Setup ---
def setup(rank, world_size):
    store_dir = os.path.join(os.getcwd(), "tmp_filestore")
    os.makedirs(store_dir, exist_ok=True)
    store_path = os.path.join(store_dir, "filestore_sync")

    init_method = f'file://{store_path}'
    timeout_delta = timedelta(minutes=10)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timeout_delta
    )
    torch.cuda.set_device(rank)
    print(f"Rank {rank} initialized using FileStore at {store_path} with Gloo backend")


def cleanup(rank):
    dist.destroy_process_group()
    if rank == 0:
        store_dir = os.path.join(os.getcwd(), "tmp_filestore")
        if os.path.exists(store_dir):
            try:
                shutil.rmtree(store_dir)
                print(f"Rank {rank} cleaned up FileStore directory: {store_dir}")
            except OSError as e:
                print(f"Rank {rank} Error removing directory {store_dir}: {e}")

# --- Utility: Save checkpoint ---
def save_checkpoint(model_state_dict, optimizer_state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict
    }, path)
    if dist.get_rank() == 0:
        print(f"✅ Checkpoint saved to {path}")

# --- Utility: Load checkpoint for ARP continuation ---
def load_model_for_arp_start(model, checkpoint_path='checkpoints/CIFAR100_epoch34_preARP.pth'):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage) 
    state_dict = checkpoint['model_state_dict']
    # Remove incompatible keys if needed
    if 'linear.weight' in state_dict:
        state_dict.pop('linear.weight')
    if 'linear.bias' in state_dict:
        state_dict.pop('linear.bias')

    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded model state from {checkpoint_path} (potentially with adjusted output layer)")
    return model

def get_datasets(dataset_name="CIFAR100"):
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
    return train_set, test_set

def evaluate(model, dataloader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.module(images) if isinstance(model, DDP) else model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train_worker(rank, world_size, alpha, mu, dataset="CIFAR100", 
                 weight_decay=1e-2, max_epochs=100, 
                 spike_threshold=1.5, instability_std=0.2, 
                 switch_min_epoch=23, batch_size=128, lr=1e-3):

    setup(rank, world_size)
    device = rank

    num_classes = 100 if dataset == "CIFAR100" else 10
    model = WideResNet28_10(num_classes=num_classes).to(device)

    # Load from checkpoint at epoch 34
    checkpoint_path = "checkpoints/CIFAR100_epoch34_preARP.pth"
    if rank == 0 and not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}. Did you run pretrain_to_epoch34?")
        cleanup(rank)
        return

    model = load_model_for_arp_start(model, checkpoint_path=checkpoint_path)
    model = model.to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    criterion = torch.nn.CrossEntropyLoss()

    train_set, test_set = get_datasets(dataset)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, 
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=False, 
                             num_workers=0, pin_memory=True)

    # Start with AdamW or ARP? - We can start with AdamW, then forcibly switch to ARP at epoch 35
    # But in the old run, we are forcibly enabling ARP right away at 35
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    switched_to_arp = False
    cpr_triggered = False
    sgd_flush_done = False

    # Logging
    writer = None
    csv_log_path = None
    if rank == 0:
        run_id = f"adamw2arp_CPRv4_{dataset}_alpha{alpha:.2e}_mu{mu:.2e}_lr{lr:.1e}_wd{weight_decay:.1e}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = f"runs/{run_id}"
        csv_log_path = f"results/{run_id}.csv"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs("results", exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard log dir: {log_dir}")
        print(f"CSV log path: {csv_log_path}")

    loss_history, accuracy_history = [], []

    csvfile = None
    writer_csv = None
    if rank == 0:
        csvfile = open(csv_log_path, mode='w', newline='')
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow(["Epoch", "Train Loss", "Train Accuracy", "Test Accuracy", "Optimizer"])

    start_epoch = 35
    # Force ARP at epoch 35 (like your original code):
    if rank == 0:
        print(f"🔄 Resumed and forced ARP optimizer at epoch 35")
    optimizer = ARPOptimizer(model.parameters(), lr=lr, alpha=alpha, mu=mu,
                             weight_decay=weight_decay,
                             clamp_G_min=0.0, clamp_G_max=10.0)
    switched_to_arp = True

    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss, correct, total = 0.0, 0, 0

        # -----------------------
        #  Staged logic
        # -----------------------
        # 1) Switch to SGD at epoch 45
        # 2) Trigger CPR at epoch 46
        # 3) Switch back to ARP at epoch 47
        if rank == 0:
            sgd_switch = (epoch == 45 and not sgd_flush_done)
            cpr_step   = (epoch == 46 and not cpr_triggered)
            arp_switch = (epoch == 47 and sgd_flush_done and cpr_triggered)
        else:
            sgd_switch = False
            cpr_step = False
            arp_switch = False

        # Broadcast decisions to all ranks
        sgd_switch_tensor = torch.tensor([int(sgd_switch)], dtype=torch.int).to(device)
        cpr_step_tensor   = torch.tensor([int(cpr_step)],   dtype=torch.int).to(device)
        arp_switch_tensor = torch.tensor([int(arp_switch)], dtype=torch.int).to(device)

        dist.broadcast(sgd_switch_tensor, src=0)
        dist.broadcast(cpr_step_tensor, src=0)
        dist.broadcast(arp_switch_tensor, src=0)

        sgd_switch = bool(sgd_switch_tensor.item())
        cpr_step   = bool(cpr_step_tensor.item())
        arp_switch = bool(arp_switch_tensor.item())

        # If we’re about to do a forward/backward pass, we want the correct optimizer in place.
        # So do the switch before we run the training loop for this epoch:
        if sgd_switch:
            # Switch to SGD
            if rank == 0:
                print("💣 Switched to SGD at epoch 45 for gradient flush")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            sgd_flush_done = True

        if cpr_step:
            # We do CPR injection after the forward pass or at the beginning of the epoch?
            # Typically we do it "before" or "after" the epoch's updates. Let’s do it at the start of the epoch.
            if rank == 0:
                print("\n🧠 CPR Triggered at epoch 46 — resetting G_ij with noise\n")
            # Because we're currently on SGD at epoch 46, we can't directly manipulate ARP's 'G' state...
            # But let's forcibly inject chaos into param.grad or param.data. 
            # For a more ARP-like effect, let's do param.data if you want a "shock" approach:
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        # We can add small random noise to the weights or forcibly clamp them
                        # This is up to you — let's do random small noise:
                        noise = torch.rand_like(p.data) * 0.05
                        p.data.add_(noise)
            cpr_triggered = True

        if arp_switch:
            # Switch back to ARP
            optimizer = ARPOptimizer(model.parameters(), lr=lr, alpha=alpha, mu=mu,
                                     weight_decay=weight_decay,
                                     clamp_G_min=0.0, clamp_G_max=10.0)
            switched_to_arp = True
            if rank == 0:
                print("⚡ Switched back to ARP at epoch 47 after CPR")

        # -----------------------
        # Training loop
        # -----------------------
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

        # Aggregate metrics
        total_loss_tensor = torch.tensor(total_loss).to(device)
        correct_tensor = torch.tensor(correct).to(device)
        total_tensor = torch.tensor(total).to(device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            # Approx average loss
            avg_loss = total_loss_tensor.item() / total_tensor.item() * batch_size
            accuracy = 100. * correct_tensor.item() / total_tensor.item()
            loss_history.append(avg_loss)
            accuracy_history.append(accuracy)

            print(f"[Rank 0] Epoch {epoch+1}/{max_epochs} completed.")
            # Evaluate occasionally
            test_acc = None
            test_freq = 1 if not switched_to_arp else 5
            if epoch % test_freq == 0:
                test_loss, test_acc = evaluate(model, test_loader, device)
                writer.add_scalar("Loss/test", test_loss, epoch)
                writer.add_scalar("Accuracy/test", test_acc, epoch)
                print(f"  [Rank 0 Eval] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            # Log
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Accuracy/train", accuracy, epoch)

            # Track which optimizer is active
            opt_name = "SGD" if (sgd_flush_done and not switched_to_arp) else ("ARP" if switched_to_arp else "AdamW")
            writer_csv.writerow([epoch+1, avg_loss, accuracy, test_acc if test_acc is not None else "", opt_name])
            csvfile.flush()

            print(f"  [{opt_name}] Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

    if rank == 0:
        writer.close()
        if csvfile:
            csvfile.close()
        print(f"📄 Logged to {csv_log_path}")

    cleanup(rank)

def main():
    # Clean up any existing FileStore dir
    store_dir = os.path.join(os.getcwd(), "tmp_filestore")
    if os.path.exists(store_dir):
        try:
            shutil.rmtree(store_dir)
            print(f"Removed existing FileStore directory: {store_dir}")
        except OSError as e:
            print(f"Warning: Could not remove existing FileStore directory {store_dir}: {e}")

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs.")

    config = {
        "alpha": 0.015,
        "mu": 0.004,
        "dataset": "CIFAR100",
        "weight_decay": 1e-2,
        "max_epochs": 100,
        "spike_threshold": 1.5,
        "instability_std": 0.2,
        "switch_min_epoch": 23,
        "batch_size": 128,
        "lr": 1e-3
    }

    print("Starting ARP Training (v4 Staged) from Checkpoint...")
    checkpoint_path = "checkpoints/CIFAR100_epoch34_preARP.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}. Run the pretrain-to-epoch34 script first.")
        return

    mp.spawn(
        train_worker,
        args=(
            world_size,
            config["alpha"],
            config["mu"],
            config["dataset"],
            config["weight_decay"],
            config["max_epochs"],
            config["spike_threshold"],
            config["instability_std"],
            config["switch_min_epoch"],
            config["batch_size"],
            config["lr"]
        ),
        nprocs=world_size,
        join=True
    )
    print("ARP + CPR v4 Training finished.")

if __name__ == "__main__":
    main()
