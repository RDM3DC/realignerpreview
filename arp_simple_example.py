import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np

# --- Custom Optimizer (ARPPiAGradientDescent) ---
class ARPPiAGradientDescent(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.01, mu=0.001):
        defaults = dict(lr=lr, alpha=alpha, mu=mu)
        super(ARPPiAGradientDescent, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']
            mu = group['mu']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize memory if first time
                if 'prev_grad' not in state:
                    state['prev_grad'] = torch.zeros_like(grad)
                    state['G'] = torch.ones_like(grad)  # start G at 1
                    state['step'] = 0

                prev_grad = state['prev_grad']
                G = state['G']
                step_num = state['step']

                # Angular curvature (πₐ) calculation
                angle_cos = torch.clamp(
                    torch.nn.functional.cosine_similarity(grad.view(-1), prev_grad.view(-1), dim=0),
                    -1.0, 1.0
                )
                angle = torch.acos(angle_cos)
                pi_a = math.pi + 0.01 * torch.tanh(angle) + 0.01 * math.log1p(step_num + 1)
                theta = pi_a / 16

                # Curve-based rotation approximation
                rot_grad = grad * torch.cos(theta) - prev_grad * torch.sin(theta)

                # Update conductance G_{ij} using ARP law
                current_I = torch.abs(rot_grad)
                G = G + alpha * current_I - mu * G
                G = torch.clamp(G, min=1e-6, max=10.0)  # stability bounds

                # Scaled update using ARP-modified gradient
                update = -group['lr'] * G * rot_grad
                p.data += update  # gradient descent (note: += because update is negative)

                # Save memory
                state['prev_grad'] = rot_grad.clone()
                state['G'] = G.clone()
                state['step'] += 1

        return loss

# --- Simple Model ---
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- Create synthetic data ---
np.random.seed(42)
X = torch.tensor(np.random.randn(100, 10), dtype=torch.float32)
y = torch.tensor(np.random.randn(100, 1), dtype=torch.float32)

# --- Training Setup ---
model = SimpleModel()
optimizer = ARPPiAGradientDescent(model.parameters(), lr=0.01, alpha=0.01, mu=0.001)
loss_fn = nn.MSELoss()

g_logs = []
pi_logs = []

# --- Training Loop ---
for epoch in range(1, 101):  # 100 epochs
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    # --- Log G and πₐ from first layer ---
    for p in model.parameters():
        state = optimizer.state[p]
        if 'G' in state:
            g_logs.append(state['G'].abs().mean().item())
        if 'prev_grad' in state:
            angle_cos = torch.clamp(
                torch.nn.functional.cosine_similarity(p.grad.view(-1), state['prev_grad'].view(-1), dim=0),
                -1.0, 1.0
            )
            angle = torch.acos(angle_cos)
            pi_a = math.pi + 0.01 * torch.tanh(angle) + 0.01 * math.log1p(state['step'])
            pi_logs.append(pi_a)
        break  # only first param

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# --- Plot G and πₐ Evolution ---
plt.figure(figsize=(10, 6))
plt.plot(g_logs, label='Mean G (conductance)')
plt.plot(pi_logs, label='πₐ (adaptive pi)')
plt.xlabel('Step')
plt.ylabel('Value')
plt.legend()
plt.title('ARP & πₐ Evolution During Training')
plt.grid(True)
plt.savefig('c:/ML_Project/arp_pia_evolution_simple.png')
print("Training completed. Plot saved as 'arp_pia_evolution_simple.png'")
