"""
adamw_acp_curvature_example.py - Enhanced example of AdamW optimizer with Adaptive Curvature Propagation (ACP)

This example demonstrates:
1. Training a small model with AdamW and ACP
2. Curvature memory tracking and visualization with TensorBoard 
3. Monitoring of per-parameter curvature metrics
4. Advanced curvature update rules based on loss dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time
from pathlib import Path

# ---- Constants for ACP (Adaptive Curvature Propagation) ----
DELTA = 0.01      # Curvature learning rate
EPSILON = 0.001   # Curvature decay rate
L_MAX = 10.0      # Reference loss value
C_MEAN_THRESHOLD = 2.0  # Threshold for curvature mean warning

# ---- Training Settings ----
NUM_EPOCHS = 50
BATCH_SIZE = 8
LOG_INTERVAL = 5

# ---- Model Definition ----
class EnhancedModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
        # Register curvature memory buffers for each layer
        c_init1 = torch.ones_like(self.layer1.weight)
        c_init2 = torch.ones_like(self.layer2.weight)
        c_init3 = torch.ones_like(self.layer3.weight)
        
        self.register_buffer("C1", c_init1)
        self.register_buffer("C2", c_init2)
        self.register_buffer("C3", c_init3)
        
        # For tracking activations
        self.g_init1 = torch.zeros_like(self.layer1.weight)
        self.g_init2 = torch.zeros_like(self.layer2.weight)
        self.g_init3 = torch.zeros_like(self.layer3.weight)
        
        self.register_buffer("G1", self.g_init1)
        self.register_buffer("G2", self.g_init2)
        self.register_buffer("G3", self.g_init3)
        
    def forward(self, x):
        # Forward pass with activation tracking
        x1 = F.relu(self.layer1(x))
        self.act1 = x1.detach().abs().mean()
        
        x2 = F.relu(self.layer2(x1))
        self.act2 = x2.detach().abs().mean()
        
        out = self.layer3(x2)
        self.act3 = out.detach().abs().mean()
        
        return out
    
    def get_curvature_tensors(self):
        return [self.C1, self.C2, self.C3]
    
    def get_activity_tensors(self):
        return [self.G1, self.G2, self.G3]
    
    def get_activation_means(self):
        # Only valid after a forward pass
        return [self.act1, self.act2, self.act3]

def compute_loss(output, target):
    return F.mse_loss(output, target)

# ---- Update curvature memory function ----
def update_curvature_memory(model, loss, delta=DELTA, epsilon=EPSILON, l_max=L_MAX):
    C_tensors = model.get_curvature_tensors()
    
    with torch.no_grad():
        # Calculate adaptive factors based on loss
        gamma = 2.0 - (loss.item() / l_max)
        base = max(l_max - loss.item(), 0.0)
        
        # Apply curvature update rule to each tensor
        for C in C_tensors:
            curvature_update = delta * (base ** gamma) - epsilon * C
            C += curvature_update
            # Ensure curvature remains positive
            C.clamp_(min=0.1)

# ---- Update activity tracking function ----
def update_activity_tracking(model, alpha=0.01, mu=0.001):
    G_tensors = model.get_activity_tensors()
    act_means = model.get_activation_means()
    C_tensors = model.get_curvature_tensors()
    
    with torch.no_grad():
        for G, act, C in zip(G_tensors, act_means, C_tensors):
            # EMA update: G = (1-mu)*G + alpha*act*C
            G.mul_(1 - mu).add_(alpha * act * C)

# ---- Main training function ----
def main():
    # Create model and optimizer
    model = EnhancedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # TensorBoard logging setup
    base_dir = Path(__file__).resolve().parent
    log_dir = base_dir / "runs" / f"adamw_acp_enhanced_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"[TensorBoard] Log directory: {log_dir}")
    print(f"[TensorBoard] Run: tensorboard --logdir \"{log_dir}\"")
    
    # Training variables
    prev_loss = None
    last_curv_warn_step = -100  # For throttling warnings
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        # Generate synthetic data for this epoch
        num_batches = 10
        for batch in range(num_batches):
            # Synthetic data
            inputs = torch.randn(BATCH_SIZE, 10)
            targets = torch.randn(BATCH_SIZE, 1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Apply curvature-aware gradient adjustment
            C_tensors = model.get_curvature_tensors()
            with torch.no_grad():
                for idx, (name, param) in enumerate(model.named_parameters()):
                    if param.grad is not None and 'weight' in name:
                        layer_idx = int(name[5]) - 1  # Extract layer number (layer1, layer2, layer3)
                        if 0 <= layer_idx < len(C_tensors):
                            curvature_adjustment = 0.05 * C_tensors[layer_idx] * param.grad
                            param.grad += curvature_adjustment
            
            # Optimizer step
            optimizer.step()
            
            # Update curvature memory based on loss dynamics
            update_curvature_memory(model, loss)
            
            # Update activity tracking
            update_activity_tracking(model)
            
            # ---- TensorBoard Logging ----
            C_tensors = model.get_curvature_tensors()
            G_tensors = model.get_activity_tensors()
            
            # Log loss
            writer.add_scalar("Loss/train", loss.item(), global_step)
            
            # Log hyperparameters
            writer.add_scalar('Curvature/DELTA', DELTA, global_step)
            writer.add_scalar('Curvature/EPSILON', EPSILON, global_step)
            
            # Log curvature statistics for each layer
            for idx, C in enumerate(C_tensors):
                writer.add_scalar(f'Curvature/C{idx+1}_mean', C.mean().item(), global_step)
                writer.add_scalar(f'Curvature/C{idx+1}_std', C.std().item(), global_step)
                
                # Log histograms every 10 steps to avoid overhead
                if global_step % 10 == 0:
                    writer.add_histogram(f'Curvature/C{idx+1}_hist', C, global_step)
            
            # Log activity tracking (G) for each layer
            for idx, G in enumerate(G_tensors):
                writer.add_scalar(f'Activity/G{idx+1}_mean', G.mean().item(), global_step)
                writer.add_scalar(f'Activity/G{idx+1}_std', G.std().item(), global_step)
            
            # Only check for high curvature warning if we have more than one curvature tensor
            if len(C_tensors) > 0:
                # Calculate average curvature across all layers
                avg_C_mean = sum(C.mean().item() for C in C_tensors) / len(C_tensors)
                
                # Log global curvature statistics
                writer.add_scalar('Curvature/global_mean', avg_C_mean, global_step)
                
                # Throttled warning for high curvature
                if avg_C_mean > C_MEAN_THRESHOLD:
                    if global_step - last_curv_warn_step >= LOG_INTERVAL * 2:
                        print(f"⚠️ Curvature mean high at step {global_step}: {avg_C_mean:.4f}")
                        last_curv_warn_step = global_step
            
            # Print progress at intervals
            if batch % LOG_INTERVAL == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch+1}/{num_batches} | Loss: {loss.item():.4f}")
                if len(C_tensors) > 0:
                    c_means = [f"{C.mean().item():.3f}" for C in C_tensors]
                    print(f"        C_means = [{', '.join(c_means)}]")
                writer.flush()  # Ensure metrics are written
            
            global_step += 1
    
    print("🎉 Training complete!")
    print(f"TensorBoard logs saved to: {log_dir}")
    writer.flush()
    writer.close()
    print("✅ TensorBoard logs flushed and closed")

if __name__ == "__main__":
    main()
