"""
ARPPiAGradientDescent Optimizer

This optimizer implements:
1. Activity-Regulated Plasticity (ARP) for adaptive learning rates
2. Pi-Adaptive (πₐ) angle-based rotation for better gradient direction
3. Momentum-like properties via gradient rotation

Author: Realigner Contributors
Date: May 8, 2025
"""

import torch
from torch.optim import Optimizer
import math

class ARPPiAGradientDescent(Optimizer):
    """
    ARPPiAGradientDescent: A novel optimizer that combines:
    
    1. ARP (Activity-Regulated Plasticity) - Inspired by synaptic plasticity in the brain,
       this dynamically adjusts the "conductance" (G) of each parameter based on gradient activity
    
    2. πₐ (Pi-Adaptive) - Adaptively rotates gradients based on the angular relationship
       between consecutive gradients for improved convergence
    
    Key parameters:
    - lr: learning rate
    - alpha: ARP activity coefficient (higher = more responsive to gradient changes)
    - mu: ARP decay coefficient (higher = faster decay of conductance)
    - weight_decay: L2 regularization parameter
    - beta: momentum factor for gradient averaging
    """
    
    def __init__(self, params, lr=0.01, alpha=0.01, mu=0.001, weight_decay=0.0001, beta=0.9):
        """
        Initialize ARPPiAGradientDescent optimizer.
        
        Args:
            params: iterable of parameters to optimize
            lr: learning rate
            alpha: ARP activity coefficient 
            mu: ARP decay coefficient
            weight_decay: L2 regularization parameter (default: 0.0001)
            beta: momentum factor for gradient averaging (default: 0.9)
        """
        defaults = dict(lr=lr, alpha=alpha, mu=mu, weight_decay=weight_decay, beta=beta)
        super(ARPPiAGradientDescent, self).__init__(params, defaults)
        
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']
            mu = group['mu']
            weight_decay = group['weight_decay']
            beta = group['beta']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                state = self.state[p]

                # Initialize state if first step
                if 'prev_grad' not in state:
                    state['prev_grad'] = torch.zeros_like(grad)
                    state['G'] = torch.ones_like(grad)  # initial conductance G = 1
                    state['step'] = 0
                    state['avg_grad'] = torch.zeros_like(grad)

                prev_grad = state['prev_grad']
                G = state['G']
                step_num = state['step']
                
                # Update exponential moving average of gradients
                if 'avg_grad' in state:
                    avg_grad = state['avg_grad']
                    avg_grad.mul_(beta).add_(grad, alpha=1-beta)
                else:
                    avg_grad = grad.clone()
                    state['avg_grad'] = avg_grad
                
                # Calculate angle between current and previous gradient
                # Use cosine similarity to find the angle
                angle_cos = torch.clamp(
                    torch.nn.functional.cosine_similarity(grad.view(-1), prev_grad.view(-1), dim=0),
                    -1.0, 1.0
                )
                angle = torch.acos(angle_cos)
                
                # Enhanced adaptive π calculation - more responsive to angle and steps
                # Use log-based scaling that grows slower with more steps
                step_factor = 0.01 * torch.log(torch.tensor(1.0 + step_num / 1000.0))
                angle_factor = 0.1 * torch.tanh(angle * 2)  # Scaled up angle impact
                pi_a = math.pi + angle_factor + step_factor
                
                # Calculate dynamic rotation fraction based on angle
                # Sharper angles get more rotation
                rot_fraction = 16.0  # Base divisor
                if angle > 0.5:  # If angle is large
                    rot_fraction = 12.0  # Increase rotation effect
                
                # Rotation angle - adaptive fraction of πₐ
                theta = pi_a / rot_fraction

                # Blend with momentum for rotation
                rot_grad = grad * torch.cos(theta) - prev_grad * torch.sin(theta)
                
                # Apply momentum-like behavior using avg_grad
                rot_grad = beta * rot_grad + (1-beta) * avg_grad
                
                # Apply ARP law to update conductance G with enhanced stability
                # G ← G + α|I| - μG where I is the "current" (gradient)
                current_I = torch.abs(rot_grad)
                
                # Adaptive alpha based on gradient magnitude
                adaptive_alpha = alpha * (1.0 + 0.1 * torch.log1p(current_I.mean()))
                
                # Update conductance with adaptive factors
                G = G + adaptive_alpha * current_I - mu * G
                
                # Clamp G with improved bounds
                G = torch.clamp(G, min=1e-6, max=50.0)  # Wider range for conductance

                # Learning rate warmup and decay
                effective_lr = group['lr']
                if step_num < 1000:
                    # Warm-up phase
                    effective_lr = group['lr'] * (step_num + 1) / 1000
                elif step_num > 10000:
                    # Decay phase
                    effective_lr = group['lr'] * (0.1 + 0.9 * torch.exp(torch.tensor(-0.0001 * (step_num - 10000))))
                
                # Compute update with effective learning rate
                update = -effective_lr * G * rot_grad
                
                # Apply update
                p.data += update

                # Store state for next iteration
                state['prev_grad'] = rot_grad.clone()
                state['G'] = G.clone()
                state['avg_grad'] = avg_grad
                state['step'] += 1

        return loss
