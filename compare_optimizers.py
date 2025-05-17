"""
Comparing ARPPiAGradientDescent vs ARPPiAGradientDescentPlus and SGD

This script compares the performance of:
1. Original ARPPiAGradientDescent
2. Enhanced ARPPiAGradientDescentPlus 
3. Standard SGD optimizer

on the MNIST dataset with identical neural network architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math
from arp_pia_optimizer import ARPPiAGradientDescent
from arp_pia_optimizer_plus import ARPPiAGradientDescentPlus

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Define a CNN for MNIST ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- Load Data ---
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# --- Training Function ---
def train(model, train_loader, optimizer, loss_fn, device, epochs=5, log_interval=100, optimizer_name=""):
    model.train()
    
    train_losses = []
    accuracy_logs = []
    g_logs = []
    pi_logs = [] 
    step_times = []
    
    total_batches = len(train_loader)
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Measure step time
            step_start = time.time()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Record step time
            step_times.append(time.time() - step_start)
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Accumulate batch loss
            epoch_loss += loss.item()
            
            # Log G and πₐ values for ARP optimizers
            if "ARP" in optimizer_name and hasattr(optimizer, "state"):
                for p in model.parameters():
                    if p in optimizer.state and 'G' in optimizer.state[p]:
                        state = optimizer.state[p]
                        g_logs.append(state['G'].abs().mean().item())
                        
                        if 'prev_grad' in state and p.grad is not None:
                            angle_cos = torch.clamp(
                                torch.nn.functional.cosine_similarity(
                                    p.grad.view(-1), state['prev_grad'].view(-1), dim=0),
                                -1.0, 1.0
                            )
                            angle = torch.acos(angle_cos)
                            pi_a = math.pi + 0.01 * torch.tanh(angle) + 0.01 * math.log1p(state['step'])
                            pi_logs.append(pi_a)
                        break  # Only log first parameter
            
            # Print progress
            if batch_idx % log_interval == 0:
                pct_complete = 100. * batch_idx / total_batches
                batch_accuracy = 100. * correct / total
                elapsed = time.time() - start_time
                print(f'[{optimizer_name}] Epoch {epoch}/{epochs} [{batch_idx}/{total_batches} ({pct_complete:.0f}%)] '
                      f'Loss: {loss.item():.6f} Accuracy: {batch_accuracy:.2f}% '
                      f'Elapsed: {elapsed:.1f}s')
        
        # End of epoch metrics
        epoch_loss /= len(train_loader)
        epoch_accuracy = 100. * correct / total
        train_losses.append(epoch_loss)
        accuracy_logs.append(epoch_accuracy)
        
        epoch_time = time.time() - epoch_start
        print(f'[{optimizer_name}] Epoch {epoch}/{epochs} complete: Avg. Loss: {epoch_loss:.6f}, '
              f'Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f}s')
    
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    print(f'[{optimizer_name}] Average step time: {avg_step_time*1000:.3f} ms')
    
    return {
        'losses': train_losses,
        'accuracy': accuracy_logs,
        'g_logs': g_logs,
        'pi_logs': pi_logs,
        'step_times': avg_step_time
    }

# --- Testing Function ---
def test(model, test_loader, loss_fn, device, optimizer_name=""):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\n[{optimizer_name}] Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

# --- Visualization Functions ---
def plot_comparison(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(15, 12))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for name, metrics in results.items():
        plt.plot(metrics['losses'], label=name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training accuracy
    plt.subplot(2, 2, 2)
    for name, metrics in results.items():
        plt.plot(metrics['accuracy'], label=name)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot test accuracy (bar chart)
    plt.subplot(2, 2, 3)
    names = list(results.keys())
    accuracies = [metrics['test_accuracy'] for metrics in results.values()]
    plt.bar(names, accuracies)
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy (%)')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    plt.grid(True, axis='y')
    
    # Plot step times
    plt.subplot(2, 2, 4)
    names = list(results.keys())
    times = [metrics['step_times'] * 1000 for metrics in results.values()]  # Convert to ms
    plt.bar(names, times)
    plt.title('Average Step Time')
    plt.ylabel('Time (ms)')
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f"{v:.2f} ms", ha='center')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")
    
    # Plot G and πₐ for ARP optimizers
    plt.figure(figsize=(12, 5))
    
    for name, metrics in results.items():
        if 'g_logs' in metrics and metrics['g_logs'] and 'ARP' in name:
            plt.subplot(1, 2, 1)
            plt.plot(metrics['g_logs'], label=f"{name} G")
            plt.title('Conductance (G) Evolution')
            plt.xlabel('Step')
            plt.ylabel('G Value')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(metrics['pi_logs'], label=f"{name} πₐ")
            plt.title('Pi-Adaptive (πₐ) Evolution')
            plt.xlabel('Step')
            plt.ylabel('πₐ Value')
            plt.legend()
            plt.grid(True)
    
    arp_plot_path = save_path.replace('.png', '_arp_metrics.png')
    plt.tight_layout()
    plt.savefig(arp_plot_path)
    print(f"ARP metrics plot saved to {arp_plot_path}")

# --- Run Comparison ---
def run_comparison(epochs=5, batch_size=64, save_path='results/comparison'):
    os.makedirs(save_path, exist_ok=True)
    
    train_loader, test_loader = load_data(batch_size)
    loss_fn = nn.CrossEntropyLoss()
    
    # Configure optimizers to test
    optimizers = {
        'SGD': {
            'optimizer': torch.optim.SGD,
            'kwargs': {'lr': 0.01, 'momentum': 0.9}
        },
        'ARPPiA': {
            'optimizer': ARPPiAGradientDescent,
            'kwargs': {'lr': 0.01, 'alpha': 0.01, 'mu': 0.001}
        },
        'ARPPiA+': {
            'optimizer': ARPPiAGradientDescentPlus,
            'kwargs': {'lr': 0.01, 'alpha': 0.01, 'mu': 0.001, 
                      'momentum': 0.9, 'warmup_steps': 500}
        }
    }
    
    results = {}
    
    for name, config in optimizers.items():
        print(f"\n{'='*50}\nTraining with {name}\n{'='*50}\n")
        
        # Create model
        model = CNN().to(device)
        
        # Create optimizer
        optimizer = config['optimizer'](model.parameters(), **config['kwargs'])
        
        # Train model
        train_metrics = train(model, train_loader, optimizer, loss_fn, device, 
                             epochs=epochs, optimizer_name=name)
        
        # Test model
        test_loss, test_accuracy = test(model, test_loader, loss_fn, device, optimizer_name=name)
        
        # Store results
        results[name] = {
            'losses': train_metrics['losses'],
            'accuracy': train_metrics['accuracy'],
            'g_logs': train_metrics.get('g_logs', []),
            'pi_logs': train_metrics.get('pi_logs', []),
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'step_times': train_metrics['step_times']
        }
        
        # Save model
        model_path = os.path.join(save_path, f'mnist_{name.lower().replace("+", "_plus")}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Plot comparison
    plot_comparison(results, os.path.join(save_path, 'optimizer_comparison.png'))
    
    # Print summary
    print("\nOptimizer Comparison Summary:")
    print("-" * 60)
    print(f"{'Optimizer':<10} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12} {'Step Time':<12}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<10} {metrics['losses'][-1]:<12.4f} {metrics['accuracy'][-1]:<12.2f}% "
              f"{metrics['test_loss']:<12.4f} {metrics['test_accuracy']:<12.2f}% "
              f"{metrics['step_times']*1000:<12.2f}ms")
    print("-" * 60)
    
    return results

if __name__ == "__main__":
    run_comparison(epochs=5, batch_size=64)
