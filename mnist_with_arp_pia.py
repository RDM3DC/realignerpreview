"""
MNIST Training with ARPPiAGradientDescent

This script demonstrates the ARPPiAGradientDescent optimizer's 
performance on the MNIST dataset, with visualizations of the 
adaptive conductance G and pi_a values during training.

Author: Your Name
Date: May 8, 2025
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

# --- Define a simpler MLP for MNIST ---
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Data Loading ---
def load_data(batch_size=64):
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# --- Training Function ---
def train(model, train_loader, optimizer, loss_fn, device, epochs=5, log_interval=100):
    model.train()
    
    # Logging metrics
    train_losses = []
    g_logs = []  # Conductance values
    pi_logs = []  # Pi-adaptive values
    accuracy_logs = []
    
    total_batches = len(train_loader)
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Accumulate batch loss
            epoch_loss += loss.item()
            
            # Log G and πₐ values from the first layer
            for p in model.parameters():
                state = optimizer.state[p]
                if 'G' in state and 'prev_grad' in state:
                    # Log G (conductance)
                    g_logs.append(state['G'].abs().mean().item())
                      # Calculate pi_a with improved formula
                    angle_cos = torch.clamp(
                        torch.nn.functional.cosine_similarity(
                            p.grad.view(-1), state['prev_grad'].view(-1), dim=0),
                        -1.0, 1.0
                    )
                    angle = torch.acos(angle_cos)
                    
                    # Enhanced pi_a calculation to match the optimizer
                    step_num = state['step']
                    step_factor = 0.01 * torch.log(torch.tensor(1.0 + step_num / 1000.0))
                    angle_factor = 0.1 * torch.tanh(angle * 2)
                    pi_a = torch.tensor(math.pi + angle_factor + step_factor)
                    
                    pi_logs.append(pi_a.item())
                    break  # Only use first parameter
            
            # Print progress
            if batch_idx % log_interval == 0:
                pct_complete = 100. * batch_idx / total_batches
                batch_accuracy = 100. * correct / total
                elapsed = time.time() - start_time
                print(f'Epoch {epoch}/{epochs} [{batch_idx}/{total_batches} ({pct_complete:.0f}%)] '
                      f'Loss: {loss.item():.6f} Accuracy: {batch_accuracy:.2f}% '
                      f'Elapsed: {elapsed:.1f}s')
        
        # End of epoch metrics
        epoch_loss /= len(train_loader)
        epoch_accuracy = 100. * correct / total
        train_losses.append(epoch_loss)
        accuracy_logs.append(epoch_accuracy)
        
        print(f'Epoch {epoch}/{epochs} complete: Avg. Loss: {epoch_loss:.6f}, '
              f'Accuracy: {epoch_accuracy:.2f}%')
    
    return train_losses, accuracy_logs, g_logs, pi_logs

# --- Testing Function ---
def test(model, test_loader, loss_fn, device):
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
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

# --- Visualization Functions ---
def plot_training_metrics(train_losses, accuracy_logs, g_logs, pi_logs, save_path):
    # Create a directory for saving plots if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Plot loss and accuracy
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(accuracy_logs)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    # Plot G and πₐ evolution
    plt.subplot(2, 2, 3)
    plt.plot(g_logs)
    plt.title('Mean Conductance (G) Evolution')
    plt.xlabel('Step')
    plt.ylabel('G Value')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(pi_logs)
    plt.title('Pi-Adaptive (πₐ) Evolution')
    plt.xlabel('Step')
    plt.ylabel('πₐ Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")
    
    # Create a separate plot for G and πₐ correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(g_logs, pi_logs, alpha=0.5, s=3)
    plt.title('Correlation between G and πₐ')
    plt.xlabel('Conductance G')
    plt.ylabel('Pi-Adaptive πₐ')
    plt.grid(True)
    correlation_path = save_path.replace('.png', '_correlation.png')
    plt.savefig(correlation_path)
    print(f"Correlation plot saved to {correlation_path}")

# --- Compare with SGD ---
def compare_optimizers(model_class, train_loader, test_loader, device, 
                       epochs=5, batch_size=64, save_path='results'):
    # Create models
    arp_model = model_class().to(device)
    sgd_model = model_class().to(device)
    
    # Create optimizers
    arp_optimizer = ARPPiAGradientDescent(
        arp_model.parameters(), 
        lr=0.01,           # Base learning rate
        alpha=0.01,        # ARP activity coefficient
        mu=0.001,          # ARP decay coefficient
        weight_decay=0.0001, # Weight decay for regularization
        beta=0.9           # Momentum factor
    )
    sgd_optimizer = torch.optim.SGD(
        sgd_model.parameters(), 
        lr=0.01, 
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Train models
    print("Training with ARPPiAGradientDescent...")
    arp_train_losses, arp_accuracy, g_logs, pi_logs = train(
        arp_model, train_loader, arp_optimizer, loss_fn, device, epochs
    )
    
    print("\nTraining with SGD...")
    sgd_train_losses, sgd_accuracy, _, _ = train(
        sgd_model, train_loader, sgd_optimizer, loss_fn, device, epochs
    )
    
    # Test models
    print("\nTesting ARPPiAGradientDescent model...")
    arp_test_loss, arp_test_acc = test(arp_model, test_loader, loss_fn, device)
    
    print("Testing SGD model...")
    sgd_test_loss, sgd_test_acc = test(sgd_model, test_loader, loss_fn, device)
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(arp_train_losses, label='ARPPiA')
    plt.plot(sgd_train_losses, label='SGD')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(arp_accuracy, label='ARPPiA')
    plt.plot(sgd_accuracy, label='SGD')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(save_path, exist_ok=True)
    comparison_path = os.path.join(save_path, 'optimizer_comparison.png')
    plt.tight_layout()
    plt.savefig(comparison_path)
    print(f"Comparison plot saved to {comparison_path}")
    
    # Plot G and πₐ evolution
    plot_training_metrics(
        arp_train_losses, arp_accuracy, g_logs, pi_logs,
        os.path.join(save_path, 'arp_pia_metrics.png')
    )
    
    # Return test metrics for both optimizers
    return {
        'ARPPiA': {'loss': arp_test_loss, 'accuracy': arp_test_acc},
        'SGD': {'loss': sgd_test_loss, 'accuracy': sgd_test_acc}
    }

# --- Main execution ---
if __name__ == "__main__":
    print("ARPPiAGradientDescent MNIST Example")
    print("-" * 40)
    
    # Parameters
    BATCH_SIZE = 64
    EPOCHS = 5
    SAVE_PATH = os.path.join('results', 'mnist')
    
    # Load data
    train_loader, test_loader = load_data(BATCH_SIZE)
    
    # Run comparison experiment
    results = compare_optimizers(
        MLP, train_loader, test_loader, device, 
        epochs=EPOCHS, batch_size=BATCH_SIZE, save_path=SAVE_PATH
    )
    
    # Print summary
    print("\nOptimizer Comparison Summary:")
    print("-" * 40)
    for name, metrics in results.items():
        print(f"{name}: Test Loss: {metrics['loss']:.4f}, Test Accuracy: {metrics['accuracy']:.2f}%")
    print("-" * 40)
