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
    # Create a temporary file for FileStore
    # Ensure the directory exists and is accessible by all processes
    store_dir = os.path.join(os.getcwd(), "tmp_filestore")
    os.makedirs(store_dir, exist_ok=True)
    store_path = os.path.join(store_dir, "filestore_sync")

    # Use file system store for rendezvous on Windows
    init_method = f'file://{store_path}'

    # Initialize the process group using FileStore and gloo backend
    # Gloo is generally better supported on Windows than NCCL
    # Add an explicit timeout (e.g., 10 minutes)
    timeout_delta = timedelta(minutes=10)
    dist.init_process_group(backend="gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timeout_delta)
    torch.cuda.set_device(rank)
    print(f"Rank {rank} initialized using FileStore at {store_path} with Gloo backend")


def cleanup(rank):
    dist.destroy_process_group()
    # Clean up the temporary file store directory (only rank 0)
    if rank == 0:
        store_dir = os.path.join(os.getcwd(), "tmp_filestore")
        if os.path.exists(store_dir):
            try:
                shutil.rmtree(store_dir)
                print(f"Rank {rank} cleaned up FileStore directory: {store_dir}")
            except OSError as e:
                print(f"Rank {rank} Error removing directory {store_dir}: {e}")

# --- Utility: Save checkpoint ---
# Modified to handle DDP model by saving model.module.state_dict()
def save_checkpoint(model_state_dict, optimizer_state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict
    }, path)
    # Print only from rank 0
    if dist.get_rank() == 0:
        print(f"✅ Checkpoint saved to {path}")

# --- Utility: Load checkpoint for ARP continuation ---
# No changes needed here as it loads state *before* DDP wrapping
def load_model_for_arp_start(model, checkpoint_path='checkpoints/adamw_epoch20.pth'):
    # Load checkpoint while ignoring mismatched layers
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage) # Load to CPU first
    state_dict = checkpoint['model_state_dict']

    # Remove incompatible keys for the output layer if necessary (e.g., changing num_classes)
    # Check if the keys exist before popping
    if 'linear.weight' in state_dict:
        state_dict.pop('linear.weight')
    if 'linear.bias' in state_dict:
        state_dict.pop('linear.bias')

    # Load the remaining state_dict
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded model state from {checkpoint_path} (potentially with adjusted output layer)")
    return model

# --- Dataset provider ---
# Modified to return datasets instead of dataloaders
def get_datasets(dataset_name="CIFAR10"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), # CIFAR10 stats
    ])
    # Adjust normalization if using CIFAR100 later
    if dataset_name == "CIFAR100":
        # Note: Using CIFAR10 normalization stats for now, update if needed
        train_set = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
    else: # Default to CIFAR10
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    return train_set, test_set

# --- Evaluation function ---
# Runs on a single process (rank 0)
def evaluate(model, dataloader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # If model is DDP, access the underlying model using .module
            outputs = model.module(images) if isinstance(model, DDP) else model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# --- DDP Training Worker ---
def train_worker(rank, world_size, alpha, mu, dataset="CIFAR10", weight_decay=1e-4, max_epochs=30, spike_threshold=1.5, instability_std=0.2, switch_min_epoch=23, batch_size=128, lr=1e-3):
    setup(rank, world_size)
    device = rank # Use rank as device ID

    num_classes = 100 if dataset == "CIFAR100" else 10
    model = WideResNet28_10(num_classes=num_classes).to(device)

    # Checkpoint handling (only rank 0 checks existence, all ranks load)
    checkpoint_path = "checkpoints/CIFAR100_epoch34_preARP.pth"
    if rank == 0 and not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}. Pre-training required.")
        print("Please run the pre-training script first.")
        cleanup(rank)
        return # Or raise an error

    # Load the checkpoint (all ranks load the same checkpoint)
    # Ensure model is on the correct device *before* loading state dict if map_location isn't used effectively
    model = load_model_for_arp_start(model, checkpoint_path=checkpoint_path)
    model = model.to(device) # Ensure model is on the correct device after loading

    # Wrap model with DDP *after* loading state dict
    model = DDP(model, device_ids=[rank], find_unused_parameters=False) # Set find_unused_parameters to False based on warning

    criterion = torch.nn.CrossEntropyLoss()

    # Get datasets and create DDP dataloaders
    train_set, test_set = get_datasets(dataset)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    # Effective batch size = batch_size * world_size
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True) # Reduced num_workers to 0
    # Test loader typically doesn't need sampling, evaluate on rank 0
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True) # Reduced num_workers to 0

    # Optimizer setup (AdamW initially)
    # Pass model.parameters() - DDP handles it
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    switched = False

    # Logging setup (only rank 0)
    writer = None
    csv_log_path = None
    if rank == 0:
        run_id = f"adamw2arp_{dataset}_alpha{alpha:.2e}_mu{mu:.2e}_lr{lr:.1e}_wd{weight_decay:.1e}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = f"runs/{run_id}"
        csv_log_path = f"results/{run_id}.csv"
        os.makedirs(f"runs/{run_id}", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard log dir: {log_dir}")
        print(f"CSV log path: {csv_log_path}")

    loss_history, accuracy_history = [], [] # Tracked locally, only used by rank 0 for decisions/logging

    csvfile = None
    writer_csv = None
    if rank == 0:
        csvfile = open(csv_log_path, mode='w', newline='')
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow(["Epoch", "Train Loss", "Train Accuracy", "Test Accuracy", "Switched to ARP"])

    start_epoch = 35
    for epoch in range(start_epoch, max_epochs):
        # --- CPRv7.3: Dual-Sensor Mode (Slope + G Drift) ---
        if epoch >= 38 and rank == 0 and 'G_control_points' in locals():
            # Slope check
            if len(loss_history) >= 3:
                expected_drop_per_epoch = 0.01
                actual_slope = (loss_history[-3] - loss_history[-1]) / 2
                slope_trigger = abs(actual_slope - expected_drop_per_epoch) > 0.008
            else:
                slope_trigger = False

            # G drift check
            total_drift = 0.0
            count = 0
            for p in model.parameters():
                if 'G' in optimizer.state[p] and id(p) in G_control_points:
                    drift = (optimizer.state[p]['G'] - G_control_points[id(p)]).abs().mean().item()
                    total_drift += drift
                    count += 1
            avg_drift = total_drift / max(1, count)
            drift_trigger = avg_drift > 0.15

            # If both triggers are hot, fire CPR
            if slope_trigger and drift_trigger:
                print(f"📏 Slope deviation: {actual_slope:.4f} | 🎯 G drift: {avg_drift:.4f}")
                print("🧭 CPRv7.3 triggered — dual condition met (slope + drift)")

                with torch.no_grad():
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                grad = p.grad.detach().abs()
                                noise = torch.rand_like(grad) * 0.05
                                state = optimizer.state[p]
                                state['G'] = (alpha / mu) * (grad + noise).clamp(min=1e-4, max=100.0)

        # --- ARP Observer Mode: Pre-collapse G memory accumulation ---
        if epoch == 25 and rank == 0:
            if 'G_control_points' not in locals():
                G_control_points = {}

            for p in model.parameters():
                if p.grad is not None:
                    grad = p.grad.detach().abs()
                    if 'G' in optimizer.state[p]:
                        G_control = (alpha / mu) * grad.clamp(min=1e-5, max=10.0)
                        optimizer.state[p]['G'] = G_control
                        G_control_points[id(p)] = G_control.clone()

            print("📌 ARP observer mode active at epoch 25 — G control points marked.")

        # --- Re-localization via Memory Deviation Check ---
        if epoch >= 38 and rank == 0 and 'G_baseline' in locals():
            total_drift = 0.0
            count = 0
            for p in model.parameters():
                if 'G' in optimizer.state[p] and id(p) in G_baseline:
                    drift = (optimizer.state[p]['G'] - G_baseline[id(p)]).abs().mean().item()
                    total_drift += drift
                    count += 1

            avg_drift = total_drift / max(1, count)

            if avg_drift > 0.15:
                print(f"📍 Drift from control point exceeded at epoch {epoch} — avg G drift: {avg_drift:.4f}")
                print("🧭 Auto re-localization triggered")

                with torch.no_grad():
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                grad = p.grad.detach().abs()
                                noise = torch.rand_like(grad) * 0.05
                                state = optimizer.state[p]
                                state['G'] = (alpha / mu) * (grad + noise).clamp(min=1e-4, max=100.0)

        if epoch == 25 and rank == 0:
            G_baseline = {id(p): optimizer.state[p]['G'].clone() for p in model.parameters() if 'G' in optimizer.state[p]}
            baseline_loss = loss_history[-1] if loss_history else None
            print("📌 G baseline and loss benchmark captured at epoch 25")

        # --- Slope-Based CPR Logic (2:1 grade check) ---
        if epoch >= 38 and len(loss_history) >= 3:
            expected_drop_per_epoch = 0.01
            actual_slope = (loss_history[-3] - loss_history[-1]) / 2

            if abs(actual_slope - expected_drop_per_epoch) > 0.008:
                if rank == 0:
                    print(f"📏 Slope check failed at epoch {epoch}: actual_slope={actual_slope:.4f} (target=~{expected_drop_per_epoch})")
                    print("🧭 CPR triggered due to loss slope deviation")

                with torch.no_grad():
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                grad = p.grad.detach().abs()
                                noise = torch.rand_like(grad) * 0.05
                                state = optimizer.state[p]
                                state['G'] = (alpha / mu) * (grad + noise).clamp(min=1e-4, max=100.0)

        # --- CPR v5 Adaptive Recovery Phase ---
        if epoch == 45:
            # Analyze last 3 accuracy deltas to decide recovery optimizer
            acc_deltas = [accuracy_history[-i] - accuracy_history[-i-1] for i in range(1, 4) if len(accuracy_history) >= i + 1]
            delta_avg = sum(acc_deltas) / len(acc_deltas) if acc_deltas else 0.0

            if delta_avg < -0.1:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
                print("💣 Using SGD at epoch 45 — sharp collapse detected")
            else:
                optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                print("🧘 Using AdamW at epoch 45 — mild instability detected")

        if epoch == 46:
            if rank == 0:
                print("\n🧠 CPR Triggered at epoch 46 — resetting G_ij with noise\n")
            with torch.no_grad():
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            grad = p.grad.detach().abs()
                            noise = torch.rand_like(grad) * 0.05
                            state = optimizer.state[p]
                            state['G'] = (alpha / mu) * (grad + noise).clamp(min=1e-4, max=100.0)

        if epoch == 47:
            optimizer = ARPOptimizer(model.parameters(),
                                     lr=lr,
                                     alpha=alpha,
                                     mu=mu,
                                     weight_decay=weight_decay,
                                     clamp_G_min=0.0,
                                     clamp_G_max=10.0)
            switched = True
            if rank == 0:
                print("⚡ Switched back to ARP at epoch 47 after CPR")

        if epoch == 35:
            optimizer = ARPOptimizer(model.parameters(),
                                     lr=lr,
                                     alpha=alpha,
                                     mu=mu,
                                     weight_decay=weight_decay,
                                     clamp_G_min=0.0,
                                     clamp_G_max=10.0)
            switched = True
            if rank == 0:
                print("🔄 Resumed and forced ARP optimizer at epoch 35")

        model.train()
        train_sampler.set_epoch(epoch) # Important for shuffling with DDP
        total_loss, correct, total = 0.0, 0, 0
        grad_norm = 0.0 # Tracked locally per epoch

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images) # DDP handles forward/backward
            loss = criterion(outputs, labels)
            loss.backward() # DDP handles gradient synchronization

            # Calculate grad norm before optimizer step (optional, potentially expensive)
            # Note: This norm is local to the rank. For global norm, need all_reduce.
            # current_grad_norm = sum(p.grad.norm().item()**2 for p in model.module.parameters() if p.grad is not None) ** 0.5
            # grad_norm += current_grad_norm

            optimizer.step()

            # Accumulate stats locally
            total_loss += loss.item() # loss is already averaged over batch size by CrossEntropyLoss
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

        # Aggregate metrics across all GPUs using dist.all_reduce
        # Sum loss and counts, then divide for average
        total_loss_tensor = torch.tensor(total_loss).to(device)
        correct_tensor = torch.tensor(correct).to(device)
        total_tensor = torch.tensor(total).to(device)
        # grad_norm_tensor = torch.tensor(grad_norm).to(device) # If tracking global norm

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        # dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.SUM) # If tracking global norm

        # Calculate average metrics (on rank 0 for logging/decisions)
        if rank == 0:
            # Divide summed loss by total number of batches * world_size OR use total samples
            # Using total samples is more robust if batch sizes vary
            avg_loss = total_loss_tensor.item() / total_tensor.item() * batch_size # Approximate avg loss per batch item
            accuracy = 100. * correct_tensor.item() / total_tensor.item()
            # avg_grad_norm = grad_norm_tensor.item() / (len(train_loader) * world_size) # Avg global norm

            loss_history.append(avg_loss)
            accuracy_history.append(accuracy)

            # --- Logging and Evaluation (Rank 0 Only) ---
            print(f"[Rank {rank}] Epoch {epoch+1}/{max_epochs} completed.") # Basic progress on all ranks

            # Smoothness metrics
            if len(loss_history) >= 5:
                loss_std = np.std(loss_history[-5:])
                writer.add_scalar("Smoothness/Loss Std (5)", loss_std, epoch)
            if epoch > 0 and len(loss_history) > 1:
                delta_loss = abs(loss_history[-1] - loss_history[-2])
                writer.add_scalar("Smoothness/Delta Loss", delta_loss, epoch)
            if len(accuracy_history) >= 5:
                 acc_std = np.std(accuracy_history[-5:])
                 writer.add_scalar("Smoothness/Accuracy Std (5)", acc_std, epoch)

            # Gradient tracking (using local approximation or averaged global norm if calculated)
            # writer.add_scalar("Gradients/Mean Norm", avg_grad_norm, epoch) # Log avg global norm

            # Test eval schedule
            test_freq = 1 if not switched else 5
            test_acc = None
            if epoch % test_freq == 0:
                test_loss, test_acc = evaluate(model, test_loader, device) # Evaluate using rank 0's model
                writer.add_scalar("Loss/test", test_loss, epoch)
                writer.add_scalar("Accuracy/test", test_acc, epoch)
                print(f"  [Rank 0 Eval] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Accuracy/train", accuracy, epoch)
            writer_csv.writerow([epoch + 1, avg_loss, accuracy, test_acc if test_acc is not None else '', int(switched)])
            csvfile.flush() # Ensure data is written

            optimizer_name = "ARP" if switched else "AdamW"
            arp_params = f" α={alpha}, μ={mu}" if switched else ""
            print(f"  [{optimizer_name}{arp_params}] Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

            # --- Optimizer Switching Logic (Decision on Rank 0) ---
            switch_decision = False
            abort_decision = False
            if not switched and epoch >= switch_min_epoch:
                delta_acc = accuracy - accuracy_history[epoch - 1] if epoch > 0 and len(accuracy_history) > 1 else 0
                loss_std = np.std(loss_history[-3:]) if len(loss_history) >= 3 else 0

                if delta_acc >= spike_threshold:
                    print("\n🔥 Spike detected — switching to ARP optimizer\n")
                    switch_decision = True
                elif loss_std > instability_std:
                    print("\n⚠️ Instability detected — aborting run\n")
                    abort_decision = True
            if epoch == 35: # Definite switch at epoch 35
                switch_decision = True
                print("\n🔄 Forced switch to ARP optimizer at epoch 35\n")

            # Broadcast decision to all ranks
            switch_tensor = torch.tensor([int(switch_decision)], dtype=torch.int).to(device)
            abort_tensor = torch.tensor([int(abort_decision)], dtype=torch.int).to(device)
        else:
            # Receive decision from rank 0
            switch_tensor = torch.tensor([0], dtype=torch.int).to(device)
            abort_tensor = torch.tensor([0], dtype=torch.int).to(device)

        dist.broadcast(switch_tensor, src=0)
        dist.broadcast(abort_tensor, src=0)

        # All ranks act on the decision
        if abort_tensor.item() == 1:
            break # Exit epoch loop on all ranks

        if not switched and switch_tensor.item() == 1:
            # All ranks switch optimizer simultaneously
            optimizer = ARPOptimizer(model.parameters(), # DDP handles parameters
                                     lr=lr,
                                     alpha=alpha,
                                     mu=mu,
                                     weight_decay=weight_decay,
                                     clamp_G_min=0.0,
                                     clamp_G_max=10.0)
            switched = True
            if rank == 0:
                 print("Optimizer switched to ARP on all ranks.")


    # --- Cleanup ---
    if rank == 0:
        writer.close()
        if csvfile:
            csvfile.close()
        print(f"📄 Logged to {csv_log_path}\n")

    cleanup(rank) # Pass rank to cleanup
    # Return values are less meaningful in spawned processes, main results logged by rank 0
    # return accuracy_history, loss_history # Only rank 0 has full history


# --- Pre-training function (DDP version) ---
def pretrain_worker(rank, world_size, dataset="CIFAR10", max_epochs=20, batch_size=128, lr=1e-3, weight_decay=1e-2):
    setup(rank, world_size)
    device = rank

    num_classes = 100 if dataset == "CIFAR100" else 10
    model = WideResNet28_10(num_classes=num_classes).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False) # Set find_unused_parameters to False based on warning

    criterion = torch.nn.CrossEntropyLoss()

    train_set, _ = get_datasets(dataset) # Only need training set
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True) # Reduced num_workers to 0

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    writer = None
    if rank == 0:
        run_id = f"adamw_pretrain_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = f"runs/{run_id}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"\n=== Pre-training {dataset} for {max_epochs} epochs (DDP) ===\n")
        print(f"TensorBoard log dir: {log_dir}")


    start_epoch = 35
    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
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

        # Aggregate metrics across GPUs
        total_loss_tensor = torch.tensor(total_loss).to(device)
        correct_tensor = torch.tensor(correct).to(device)
        total_tensor = torch.tensor(total).to(device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_loss = total_loss_tensor.item() / total_tensor.item() * batch_size # Approx avg loss per item
            accuracy = 100. * correct_tensor.item() / total_tensor.item()

            writer.add_scalar("Pretrain/Loss/train", avg_loss, epoch)
            writer.add_scalar("Pretrain/Accuracy/train", accuracy, epoch)
            print(f"[AdamW Pretrain] Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

            # Save the checkpoint ONLY at the specified epoch (e.g., 20)
            if epoch + 1 == 20: # Hardcoded epoch 20 for saving
                checkpoint_path = "checkpoints/CIFAR100_epoch34_preARP.pth"
                # Save the underlying model's state dict
                save_checkpoint(model.module.state_dict(), optimizer.state_dict(), checkpoint_path)
                print(f"✅ Saved pre-train checkpoint for {dataset} at epoch 20: {checkpoint_path}")

    # Save final checkpoint if max_epochs is different from 20
    if rank == 0 and max_epochs != 20:
        checkpoint_path = f"checkpoints/{dataset}_epoch{max_epochs}.pth"
        save_checkpoint(model.module.state_dict(), optimizer.state_dict(), checkpoint_path)
        print(f"✅ Saved final pre-train checkpoint for {dataset} at epoch {max_epochs}: {checkpoint_path}")


    if rank == 0:
        writer.close()
        print(f"📄 Pre-training completed for {dataset}. Checkpoints saved.")

    cleanup(rank) # Pass rank to cleanup


# --- Main Function to Spawn Processes ---
def main():
    # --- Clean up previous FileStore directory before starting ---
    store_dir = os.path.join(os.getcwd(), "tmp_filestore")
    if os.path.exists(store_dir):
        try:
            shutil.rmtree(store_dir)
            print(f"Removed existing FileStore directory: {store_dir}")
        except OSError as e:
            print(f"Warning: Could not remove existing FileStore directory {store_dir}: {e}")
    # -----------------------------------------------------------

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs.")

    # --- Configuration ---
    config = {
        "alpha": 0.015,
        "mu": 0.004,
        "dataset": "CIFAR100", # Switch dataset here
        "weight_decay": 1e-2, # Match AdamW WD? Or keep 1e-4? Using 1e-2 based on filename convention
        "max_epochs": 100,
        "spike_threshold": 1.5,
        "instability_std": 0.2,
        "switch_min_epoch": 23, # Lowered from 30
        "batch_size": 128, # Per GPU batch size
        "lr": 1e-3,
        "pretrain_epochs": 20 # Epoch to save pretrain checkpoint
    }

    # --- Choose Action ---
    # Set to True to run pre-training, False to run ARP training from checkpoint
    RUN_PRETRAINING = False # <-- Set this to False to skip pre-training and start from checkpoint

    if RUN_PRETRAINING:
        print("Starting Pre-training...")
        mp.spawn(pretrain_worker,
                 args=(world_size, config["dataset"], config["pretrain_epochs"], config["batch_size"], config["lr"], config["weight_decay"]),
                 nprocs=world_size,
                 join=True)
        print("Pre-training finished.")
    else:
        print("Starting ARP Training from Checkpoint...")
        # Check if checkpoint exists before spawning
        checkpoint_path = f"checkpoints/{config['dataset']}_epoch{config['pretrain_epochs']}.pth"
        if not os.path.exists(checkpoint_path):
             print(f"❌ Checkpoint not found at {checkpoint_path}. Run pre-training first by setting RUN_PRETRAINING = True.")
             return

        mp.spawn(train_worker,
                 args=(world_size, config["alpha"], config["mu"], config["dataset"], config["weight_decay"],
                       config["max_epochs"], config["spike_threshold"], config["instability_std"],
                       config["switch_min_epoch"], config["batch_size"], config["lr"]),
                 nprocs=world_size,
                 join=True)
        print("ARP Training finished.")


if __name__ == "__main__":
    # Note: Ensure the script is run using: torchrun --nproc_per_node=NUM_GPUS train_with_batch_runner_smoothness.py
    # Or: python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train_with_batch_runner_smoothness.py
    main()