"""
ARPPiAGradientDescent+ Optimizer

An enhanced version of ARPPiAGradientDescent with:
1. Adaptive rotation factor
2. Warmup period
3. Momentum integration
4. Adaptive decay

Author: Your Name
Date: May 8, 2025
"""

import torch
from torch.optim import Optimizer
import math

class ARPPiAGradientDescentPlus(Optimizer):
    """
    ARPPiAGradientDescent+: Enhanced version of ARPPiAGradientDescent
    
    Key enhancements:
    - Adaptive rotation factor based on training progress
    - Warmup period for conductance values
    - Momentum integration
    - Adaptive decay based on gradient history
    """
    
    def __init__(self, params, lr=0.01, alpha=0.01, mu=0.001, 
                 momentum=0.9, warmup_steps=500, 
                 adaptive_rotation=True, adaptive_decay=True):
        """
        Initialize ARPPiAGradientDescentPlus optimizer.
        
        Args:
            params: iterable of parameters to optimize
            lr: learning rate
            alpha: ARP activity coefficient 
            mu: ARP decay coefficient
            momentum: momentum factor
            warmup_steps: number of steps for G warmup
            adaptive_rotation: whether to adapt rotation factor
            adaptive_decay: whether to adapt decay coefficient
        """
        defaults = dict(lr=lr, alpha=alpha, mu=mu, 
                        momentum=momentum, warmup_steps=warmup_steps,
                        adaptive_rotation=adaptive_rotation,
                        adaptive_decay=adaptive_decay)
        super(ARPPiAGradientDescentPlus, self).__init__(params, defaults)
        
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
            momentum = group['momentum']
            warmup_steps = group['warmup_steps']
            adaptive_rotation = group['adaptive_rotation']
            adaptive_decay = group['adaptive_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state if first step
                if len(state) == 0:
                    state['prev_grad'] = torch.zeros_like(grad)
                    state['G'] = torch.ones_like(grad)  # initial conductance G = 1
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(grad)
                    state['grad_var'] = torch.zeros_like(grad)
                    state['grad_mean'] = torch.zeros_like(grad)

                prev_grad = state['prev_grad']
                G = state['G']
                step_num = state['step']
                momentum_buffer = state['momentum_buffer']
                
                # Update exponential moving average of gradient
                if 'grad_mean' in state:
                    state['grad_mean'] = 0.9 * state['grad_mean'] + 0.1 * grad
                    state['grad_var'] = 0.9 * state['grad_var'] + 0.1 * (grad - state['grad_mean']).pow(2)

                # Calculate angle between current and previous gradient
                angle_cos = torch.clamp(
                    torch.nn.functional.cosine_similarity(grad.view(-1), prev_grad.view(-1), dim=0),
                    -1.0, 1.0
                )
                angle = torch.acos(angle_cos)
                
                # Adaptive π calculation
                pi_a = math.pi + 0.01 * torch.tanh(angle) + 0.01 * math.log1p(step_num + 1)
                
                # Adaptive rotation factor based on training progress
                if adaptive_rotation:
                    # Decrease rotation as training progresses for stability
                    rotation_factor = 16.0 * (1.0 + step_num / 10000) 
                else:
                    rotation_factor = 16.0
                
                theta = pi_a / rotation_factor

                # Incorporate momentum into rotation
                momentum_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                # Enhanced rotation with momentum
                rot_grad = grad * torch.cos(theta) - prev_grad * torch.sin(theta)
                rot_grad = rot_grad.add_(momentum_buffer, alpha=momentum)
                
                # Apply warmup to G updates
                warmup_factor = min(1.0, step_num / warmup_steps) if warmup_steps > 0 else 1.0
                
                # Adaptive decay based on gradient variance
                if adaptive_decay and 'grad_var' in state:
                    # Increase decay when gradients are noisy (high variance)
                    local_mu = mu * (1.0 + 0.1 * torch.sqrt(state['grad_var']).mean().item())
                else:
                    local_mu = mu
                
                # Update conductance G_{ij} using enhanced ARP law with warmup
                current_I = torch.abs(rot_grad)
                G = G + warmup_factor * alpha * current_I - local_mu * G
                G = torch.clamp(G, min=1e-6, max=10.0)  # stability bounds

                # Scaled update using enhanced gradient
                update = -group['lr'] * G * rot_grad
                p.data.add_(update)

                # Save state
                state['prev_grad'] = rot_grad.clone()
                state['G'] = G.clone()
                state['momentum_buffer'] = momentum_buffer.clone()
                state['step'] += 1

        return loss
