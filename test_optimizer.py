import torch
import math

print("Testing ARPPiAGradientDescent Optimizer")

class ARPPiAGradientDescent(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.01, mu=0.001):
        defaults = dict(lr=lr, alpha=alpha, mu=mu)
        super(ARPPiAGradientDescent, self).__init__(params, defaults)
        print(f"Initialized optimizer with lr={lr}, alpha={alpha}, mu={mu}")

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

# Create a simple model and parameter
param = torch.nn.Parameter(torch.randn(3, 3))
print(f"Parameter shape: {param.shape}")

# Create optimizer
optimizer = ARPPiAGradientDescent([param], lr=0.01, alpha=0.01, mu=0.001)

# Simulate a gradient
param.grad = torch.randn(3, 3)
print(f"Gradient shape: {param.grad.shape}")

# Step the optimizer
optimizer.step()
print("Optimizer step completed")

# Check state
state = optimizer.state[param]
print(f"State keys: {state.keys()}")
print(f"G mean: {state['G'].mean().item()}")
print(f"Step: {state['step']}")

print("Test completed successfully")
