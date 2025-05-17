from models.simple_cnn import SimpleCNN
from models.resnet import ResNet18, WideResNet28_10
from optimizers.arp_optimizer import ARPOptimizer
from optimizers.mlarp_optimizer import MLARPOptimizer
from optimizers.arp_adamw_optimizer import ARPAdamW
from optimizers.narp_optimizer import NARPOptimizer
from optimizers.robust_mlarp_optimizer import RobustMLARP
from optimizers.unified_arp_optimizer import UnifiedARPOptimizer
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, AdamW, SGD
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Extract optimizer settings for naming
def get_optimizer_settings_name(optimizer, optimizer_type):
    if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
        group = optimizer.param_groups[0]
        alpha = group.get('alpha', None)
        mu = group.get('mu', None)
        settings = f"{optimizer_type}"
        if alpha is not None and mu is not None:
            settings += f"_alpha{alpha:.2e}_mu{mu:.2e}"
    else:
        settings = optimizer_type
    return settings

# Generate a unique folder name based on the current timestamp
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Add a parameter to select the optimizer
def train(optimizer_type='adam', epochs=50, switch_epoch=20, model_type='wide_resnet'):
    print(f"\n{'='*50}")
    print(f"TRAINING WITH {optimizer_type.upper()} OPTIMIZER AND {model_type.upper()} MODEL")
    print(f"{'='*50}\n")
    
    # Store metrics for analysis
    epoch_list = []
    loss_list = []
    accuracy_list = []
    
    # Initialize model based on model_type
    if model_type == 'simple_cnn':
        model = SimpleCNN()
    elif model_type == 'resnet18':
        model = ResNet18()
    elif model_type == 'wide_resnet':
        model = WideResNet28_10()
    else:
        raise ValueError("Unsupported model type. Choose from 'simple_cnn', 'resnet18', 'wide_resnet'.")
        
    criterion = torch.nn.CrossEntropyLoss()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # CIFAR-10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./datasets', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2470, 0.2435, 0.2616)),
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )

    # Test dataset for evaluation
    test_dataset = torchvision.datasets.CIFAR10(
        root='./datasets', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2470, 0.2435, 0.2616)),
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )

    # Initialize optimizer based on the selected type
    if optimizer_type == 'unified_arp':
        optimizer = UnifiedARPOptimizer(
            model.parameters(),
            lr=0.001,
            alpha=0.01,
            mu=0.00521,
            mode='multilayer',
            depth_factor=0.9,
            noise_threshold=0.05,
            grad_clip=1.0,
            clamp_G_min=0.0,
            clamp_G_max=10.0
        )
        print("Using UnifiedARPOptimizer with optimal default settings")
    elif optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=0.001)
        print("Using Adam optimizer for all epochs")
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        print("Using SGD optimizer for all epochs")
    elif optimizer_type == 'arp':
        # Start with AdamW for the first 20 epochs
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        print("Using AdamW optimizer for the first 20 epochs")
        
    # Extract optimizer hyperparameters safely
    group = optimizer.param_groups[0]
    alpha = group.get('alpha', None)
    mu = group.get('mu', None)
    lr = group.get('lr', None)
    weight_decay = group.get('weight_decay', None)

    # Build folder name from what's available
    folder_parts = [optimizer_type]

    # For ARP optimizers, ensure alpha and mu are included if they're expected to be there
    if optimizer_type in ['arp', 'mlarp', 'narp', 'unified_arp', 'robust_mlarp']:
        # If we're starting with AdamW, these will become available after the switch
        # so we'll provide default values for the folder name
        if alpha is None:
            alpha = 1e-2  # Default from arp_optimizer.py
            folder_parts.append(f"alpha{alpha:.2e}")
        else:
            folder_parts.append(f"alpha{alpha:.2e}")
            
        if mu is None:
            mu = 1e-3  # Default from arp_optimizer.py
            folder_parts.append(f"mu{mu:.2e}")
        else:
            folder_parts.append(f"mu{mu:.2e}")
    else:
        # For non-ARP optimizers, only include if actually present
        if alpha is not None:
            folder_parts.append(f"alpha{alpha:.2e}")
        if mu is not None:
            folder_parts.append(f"mu{mu:.2e}")
    
    if lr is not None:
        folder_parts.append(f"lr{lr:.0e}")
    if weight_decay is not None and weight_decay > 0:
        folder_parts.append(f"wd{weight_decay:.0e}")

    folder_parts.append(model_type)

    # Timestamp to ensure uniqueness - remove if not needed
    # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # folder_parts.append(current_time)

    # Final run name and writer init
    run_name = "_".join(folder_parts)
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"Logging to: runs/{run_name}")
    
    # Save configuration to JSON file for reproducibility
    import json
    
    config = {
        "optimizer": optimizer_type,
        "alpha": float(alpha) if alpha else None,
        "mu": float(mu) if mu else None,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "model_type": model_type,
        "timestamp": current_time
    }
    
    with open(f"runs/{run_name}/config.json", 'w') as f:
        json.dump(config, f, indent=4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        # Switch to ARPOptimizer after 20 epochs
        if epoch == 20 and optimizer_type == 'arp':
            optimizer = ARPOptimizer(
                model.parameters(),
                lr=1e-3,           # Good general learning rate
                weight_decay=1e-4, # Small weight decay to avoid overfitting (optional)
                clamp_G_min=0.0,   # G can't go negative
                clamp_G_max=10.0   # Keeps G stable
            )
            print("Switched to ARPOptimizer after 20 epochs")

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Use a try-except block to catch any potential issues
            try:
                optimizer.step()
            except Exception as e:
                print(f"Error during optimizer step: {e}")
                return

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        # Store metrics
        epoch_list.append(epoch + 1)
        loss_list.append(avg_loss)
        accuracy_list.append(accuracy)

        # Log to TensorBoard
        writer.add_scalar(f'Loss/{optimizer_type}', avg_loss, epoch)
        writer.add_scalar(f'Accuracy/{optimizer_type}', accuracy, epoch)

        # Print progress
        print(f'Epoch [{epoch+1}/{epochs}]: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            
            test_accuracy = 100. * test_correct / test_total
            print(f'Test Accuracy: {test_accuracy:.2f}%')
            writer.add_scalar(f'TestAccuracy/{optimizer_type}', test_accuracy, epoch)
            model.train()

    # Generate a unique name for the optimizer and its settings
    if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
        group = optimizer.param_groups[0]
        if 'alpha' in group and 'mu' in group:
            optimizer_settings = f"{optimizer_type}.{group['alpha']:.2f}a.{group['mu']:.3f}mu"
        else:
            optimizer_settings = f"{optimizer_type}"
    else:
        optimizer_settings = f"{optimizer_type}"

    # Save the trained model with the new naming convention
    model_path = f'trained_model_{optimizer_settings}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Generate performance plot with the new naming convention
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, loss_list)
    plt.title(f'Loss - {optimizer_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if optimizer_type in ['arp', 'mlarp', 'arp_adamw', 'narp', 'robust_mlarp']:
        plt.axvline(x=switch_epoch+1, color='r', linestyle='--', 
                   label=f'Switch to {optimizer_type}')
        plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, accuracy_list)
    plt.title(f'Accuracy - {optimizer_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    if optimizer_type in ['arp', 'mlarp', 'arp_adamw', 'narp', 'robust_mlarp']:
        plt.axvline(x=switch_epoch+1, color='r', linestyle='--', 
                   label=f'Switch to {optimizer_type}')
        plt.legend()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{optimizer_type}_performance.png')
    print(f"\nPerformance plot saved to results/{optimizer_type}_performance.png")
    
    return epoch_list, loss_list, accuracy_list

def analyze_results(results):
    """Analyze and compare optimizer performance."""
    plt.figure(figsize=(15, 10))
    
    # Plot loss for all optimizers
    plt.subplot(2, 1, 1)
    for opt, (epochs, loss, acc) in results.items():
        plt.plot(epochs, loss, label=opt)
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy for all optimizers
    plt.subplot(2, 1, 2)
    for opt, (epochs, loss, acc) in results.items():
        plt.plot(epochs, acc, label=opt)
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/optimizer_comparison.png')
    print("\nComparison plot saved to results/optimizer_comparison.png")
    
    # Print final metrics
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"{'Optimizer':<15} {'Final Loss':<15} {'Final Accuracy':<15}")
    print("-"*45)
    for opt, (epochs, loss, acc) in results.items():
        print(f"{opt:<15} {loss[-1]:<15.4f} {acc[-1]:<15.2f}")

if __name__ == "__main__":
    # Run only the ARP optimizer with AdamW start
    optimizer_type = 'arp'  # Set to ARP optimizer
    model_type = 'wide_resnet'  # Using WideResNet by default

    print(f"\nRunning benchmark with {optimizer_type} optimizer...")
    epochs, loss, accuracy = train(optimizer_type=optimizer_type, epochs=50, switch_epoch=20, model_type=model_type)

    # Analyze results for the single optimizer
    results = {optimizer_type: (epochs, loss, accuracy)}
    analyze_results(results)
