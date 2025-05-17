# --- AdamW Pretrain + ARP Continue Script with Watcher, CPR, Variance Tracking ---

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch
from torch.utils.tensorboard import SummaryWriter
import json  # Add missing json import

import torchvision
import torchvision.transforms as transforms

import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from models.resnet import WideResNet28_10 
from optimizers.arp_optimizer import ARPOptimizer

# Parameters
BATCH_SIZE = 512
NUM_WORKERS = 8
LEARNING_RATE = 0.003  # AdamW learning rate
PRETRAIN_EPOCHS = 20
CONTINUE_EPOCHS = 1000
ALPHA = 0.01         # ARP alpha
MU = 0.001           # ARP mu
USE_AMP = True       # Use mixed precision

# CPR parameters
CPR_TRIGGER_EPOCHS = 3
LOSS_PLATEAU_THRESHOLD = 0.005

class Watcher:
    def __init__(self):
        self.last_losses = []
        self.patience = CPR_TRIGGER_EPOCHS
        self.triggered = False

    def check_loss_plateau(self, current_loss):
        self.last_losses.append(current_loss)
        if len(self.last_losses) > self.patience:
            self.last_losses.pop(0)
        if len(self.last_losses) == self.patience:
            delta = max(self.last_losses) - min(self.last_losses)
            if delta < LOSS_PLATEAU_THRESHOLD:
                self.triggered = True
                return True
        return False

    def reset_trigger(self):
        self.triggered = False

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/realignr_{run_id}"  # ✅ Matches Watcher
    writer = SummaryWriter(log_dir)
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    # Data preparation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    # Load CIFAR-100
    print(f"Loading CIFAR-100 dataset...")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = WideResNet28_10(num_classes=100).to(device)
    model = torch.nn.DataParallel(model)
    
    # Phase 1: AdamW Pretraining
    print("\n=== Phase 1: Pretraining with AdamW ===")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
        
        acc = correct / len(trainset)
        writer.add_scalar("Pretrain/Loss", total_loss / len(trainloader), epoch)
        writer.add_scalar("Pretrain/Accuracy", acc, epoch)
        writer.flush()  # Explicitly flush data to disk
        print(f"Epoch {epoch+1}/{PRETRAIN_EPOCHS} | Loss: {total_loss/len(trainloader):.4f} | Accuracy: {acc*100:.2f}%")
    
    # Save model after AdamW pretraining
    torch.save(model.state_dict(), os.path.join("checkpoints", "adamw_epoch20.pth"))
    print("✅ AdamW Pretrain Checkpoint saved")
    
    # Phase 2: ARP Training 
    print("\n=== Phase 2: Continue Training with ARP ===")
    arp_optimizer = ARPOptimizer(model.parameters(), alpha=ALPHA, mu=MU)
    print("✅ ARP Optimizer Initialized")
    print("🔥 Warm-started G_ij from initial gradient snapshot")
    
    watcher = Watcher()
    
    # Load dataset switching functionality
    from examples.dataset_switcher_patch import (
        load_dataset, load_dataset_schedule, get_current_dataset
    )
    
    # Load training configuration if exists
    training_config = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS}
    try:
        if os.path.exists("training_config.json"):
            with open("training_config.json", "r") as f:
                training_config.update(json.load(f))
                print(f"📋 Loaded training configuration: {training_config}")
    except Exception as e:
        print(f"⚠️ Error loading training_config.json: {e}")
    
    dataset_schedule = load_dataset_schedule()
    last_dataset = None
    current_dataset = None
    
    # Initialize for dataset switching
    current_dataset = get_current_dataset(20, dataset_schedule)
    if current_dataset != "CIFAR100":  # First switch if needed
        print(f"🔄 Starting with custom dataset: {current_dataset}")
        trainset, testset = None, None  # Will be loaded on first epoch
    
    try:
        epoch = 20
        while True:  # ✅ Infinity mode
            # Dataset switching logic
            new_dataset = get_current_dataset(epoch, dataset_schedule)
            if new_dataset != current_dataset:
                print(f"🔄 Dataset switch detected: {current_dataset} → {new_dataset} at epoch {epoch}")
                
                # Save G_ij memory for current dataset
                if current_dataset:
                    g_path = os.path.join("checkpoints", f"g_memory_{current_dataset}.pt")
                    torch.save(arp_optimizer.state_dict(), g_path)
                    print(f"💾 Saved G_ij memory for {current_dataset}")
                
                # Set up new dataset
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                ])
                
                if new_dataset == "CIFAR10":
                    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
                    num_classes = 10
                elif new_dataset == "CIFAR100":
                    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
                    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
                    num_classes = 100
                elif new_dataset == "SVHN":
                    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
                    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
                    num_classes = 10
                
                trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=training_config["batch_size"], 
                    shuffle=True, 
                    num_workers=training_config["num_workers"]
                )
                testloader = torch.utils.data.DataLoader(
                    testset, 
                    batch_size=training_config["batch_size"], 
                    shuffle=False, 
                    num_workers=training_config["num_workers"]
                )
                
                # Load G_ij memory for new dataset if available
                g_path = os.path.join("checkpoints", f"g_memory_{new_dataset}.pt")
                if os.path.exists(g_path):
                    arp_optimizer.load_state_dict(torch.load(g_path))
                    print(f"🔁 Loaded G_ij memory for {new_dataset}")
                else:
                    print(f"🆕 No prior G_ij memory found for {new_dataset}, starting fresh")
                
                current_dataset = new_dataset
                print(f"✅ Dataset switched to {current_dataset}. Train samples: {len(trainset)}")
            
            model.train()
            total_loss = 0
            correct = 0
            batch_accs = []
            grad_norm = 0.0
            
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                arp_optimizer.zero_grad()
                
                with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(arp_optimizer)
                scaler.update()
                
                total_loss += loss.item()
                correct += (outputs.argmax(1) == targets).sum().item()
                batch_accs.extend((outputs.argmax(1) == targets).float().tolist())
                
                # Track gradient norm
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
            
            # Calculate metrics
            acc = correct / len(trainset)
            avg_loss = total_loss / len(trainloader)
            epoch_var = torch.tensor(batch_accs).var().item()
            
            # Log to TensorBoard
            writer.add_scalar("ARP/Loss", avg_loss, epoch)
            writer.add_scalar("ARP/Accuracy", acc, epoch)
            writer.add_scalar("ARP/AccuracyVariance", epoch_var, epoch)
            writer.add_scalar("ARP/GradNorm", grad_norm, epoch)
            writer.add_scalar("Dataset/Current", {"CIFAR10": 0, "CIFAR100": 1, "SVHN": 2}.get(current_dataset, -1), epoch)
            
            # Get alpha and mu
            alpha_val = arp_optimizer.param_groups[0]['alpha']
            mu_val = arp_optimizer.param_groups[0]['mu']
            writer.add_scalar("alpha", alpha_val, epoch)
            writer.add_scalar("mu", mu_val, epoch)
            
            # Calculate G_mean if the method exists, otherwise use a placeholder
            try:
                if hasattr(arp_optimizer, 'get_g_mean'):
                    g_mean = arp_optimizer.get_g_mean()
                    writer.add_scalar("G_mean", g_mean, epoch)
                else:
                    # Calculate average G value manually if possible
                    g_sum = 0
                    g_count = 0
                    for group in arp_optimizer.param_groups:
                        for p in group['params']:
                            state = arp_optimizer.state[p]
                            if 'G' in state:
                                g_sum += state['G'].abs().mean().item()
                                g_count += 1
                    
                    if g_count > 0:
                        writer.add_scalar("G_mean", g_sum / g_count, epoch)
                    else:
                        writer.add_scalar("G_mean", 0.0, epoch)  # Placeholder when G isn't available
            except Exception as e:
                print(f"⚠️ Could not calculate G_mean: {e}")
                writer.add_scalar("G_mean", 0.0, epoch)  # Fallback value
                
            writer.flush()  # Explicitly flush data to disk
            
            # Print progress
            print(f"[ARP] Epoch {epoch} | Dataset: {current_dataset} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}% | Var: {epoch_var:.6f} | GradNorm: {grad_norm:.2f}")
            
            # CPR Plateau Detection and Recovery
            if watcher.check_loss_plateau(avg_loss):
                print("⚠️ CPR Triggered: loss plateau detected. Adjusting alpha/mu...")
                writer.add_scalar("CPR_trigger", 1.0, epoch)
                
                for group in arp_optimizer.param_groups:
                    group['alpha'] *= 1.05  # Increase alpha
                    group['mu'] *= 0.95     # Decrease mu
                
                watcher.reset_trigger()
            else:
                writer.add_scalar("CPR_trigger", 0.0, epoch)
            
            # External control from GPT feedback
            if os.path.exists("realignr_control.json"):
                import json
                with open("realignr_control.json", "r") as f:
                    control = json.load(f)
                    for group in arp_optimizer.param_groups:
                        group["alpha"] = control["alpha"]
                        group["mu"] = control["mu"]
                    print(f"🧪 CPR Applied from feedback: alpha={control['alpha']}, mu={control['mu']}")
                os.remove("realignr_control.json")
            
            # Save checkpoints at regular intervals
            if epoch % 50 == 0:
                ckpt_path = os.path.join("checkpoints", f"realignr_v5_epoch{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': arp_optimizer.state_dict(),
                    'loss': avg_loss,
                    'dataset': current_dataset
                }, ckpt_path)
                print(f"✅ Checkpoint saved at {ckpt_path}")
            
            epoch += 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user (Ctrl+C)")
        # Save emergency checkpoint
        emergency_path = os.path.join("checkpoints", f"emergency_save_epoch{epoch}.pth")
        
        # Also save current G_ij for current dataset
        if current_dataset:
            g_path = os.path.join("checkpoints", f"g_memory_{current_dataset}.pt")
            torch.save(arp_optimizer.state_dict(), g_path)
            print(f"💾 Emergency save: G_ij memory for {current_dataset}")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': arp_optimizer.state_dict(),
            'loss': avg_loss,
            'dataset': current_dataset
        }, emergency_path)
        print(f"💾 Emergency checkpoint saved at {emergency_path}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
