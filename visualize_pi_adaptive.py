"""
Pi-Adaptive (πₐ) Gradient Rotation Visualization

This script visualizes how the πₐ parameter affects gradient rotation
by showing the effect on different gradient directions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import os

def visualize_pi_adaptive_rotation(save_dir='results/pi_adaptive'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure for static visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define parameters
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)  # 8 different gradient directions
    step_range = np.arange(0, 1000, 50)  # Different training steps
    angle_range = np.linspace(0, np.pi, 50)  # Different gradient angles
    
    # Compute pi_a for different angles and steps
    pi_a_values = np.zeros((len(angle_range), len(step_range)))
    
    for i, angle in enumerate(angle_range):
        for j, step in enumerate(step_range):
            pi_a = math.pi + 0.01 * math.tanh(angle) + 0.01 * math.log1p(step)
            pi_a_values[i, j] = pi_a
    
    # Plot pi_a as function of angle and step
    im = axs[0, 0].imshow(
        pi_a_values, 
        extent=[0, step_range[-1], 0, angle_range[-1]], 
        aspect='auto', 
        origin='lower',
        cmap='viridis'
    )
    axs[0, 0].set_title('πₐ Value by Angle and Step')
    axs[0, 0].set_xlabel('Training Step')
    axs[0, 0].set_ylabel('Gradient Angle θ (rad)')
    plt.colorbar(im, ax=axs[0, 0], label='πₐ Value')
    
    # Plot rotation effect for fixed step and varying angles
    step = 100
    angle_vals = np.linspace(0, np.pi, 100)
    pi_a_vals = [math.pi + 0.01 * math.tanh(a) + 0.01 * math.log1p(step) for a in angle_vals]
    rotation_angles = [pi_a / 16 for pi_a in pi_a_vals]
    
    axs[0, 1].plot(angle_vals, rotation_angles)
    axs[0, 1].set_title(f'Rotation Angle vs Gradient Angle (Step {step})')
    axs[0, 1].set_xlabel('Gradient Angle θ (rad)')
    axs[0, 1].set_ylabel('Rotation Angle (rad)')
    axs[0, 1].grid(True)
    
    # Visualize effect on gradient directions for early and late training
    for i, step in enumerate([10, 1000]):
        ax = axs[1, i]
        
        # Original gradients (8 directions)
        for angle in angles:
            orig_x, orig_y = math.cos(angle), math.sin(angle)
            ax.arrow(0, 0, orig_x, orig_y, head_width=0.05, head_length=0.08, 
                    fc='blue', ec='blue', alpha=0.5, label='Original' if angle == angles[0] else "")
        
        # Rotated gradients
        for angle in angles:
            # Original gradient
            grad_x, grad_y = math.cos(angle), math.sin(angle)
            
            # Previous gradient (assume orthogonal for visualization)
            prev_grad_x, prev_grad_y = -math.sin(angle), math.cos(angle)
            
            # Calculate pi_a and rotation angle
            cos_sim = grad_x * prev_grad_x + grad_y * prev_grad_y
            grad_angle = math.acos(max(-1.0, min(1.0, cos_sim)))
            pi_a = math.pi + 0.01 * math.tanh(grad_angle) + 0.01 * math.log1p(step)
            theta = pi_a / 16
            
            # Rotated gradient
            rot_x = grad_x * math.cos(theta) - prev_grad_x * math.sin(theta)
            rot_y = grad_y * math.cos(theta) - prev_grad_y * math.sin(theta)
            
            # Normalize
            norm = math.sqrt(rot_x**2 + rot_y**2)
            rot_x, rot_y = rot_x/norm, rot_y/norm
            
            ax.arrow(0, 0, rot_x, rot_y, head_width=0.05, head_length=0.08, 
                    fc='red', ec='red', alpha=0.7, label='Rotated' if angle == angles[0] else "")
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f'Gradient Rotation (Step {step})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pi_adaptive_rotation.png'))
    print(f"Rotation visualization saved to {save_dir}/pi_adaptive_rotation.png")
    
    # Create animation to show rotation effect over training
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title('Gradient Rotation Evolution During Training')
    
    # Original gradient vectors
    orig_lines = []
    for angle in angles:
        orig_x, orig_y = math.cos(angle), math.sin(angle)
        line, = ax.plot([0, orig_x], [0, orig_y], 'b-', alpha=0.3)
        orig_lines.append(line)
    
    # Rotated gradient vectors
    rot_lines = []
    for _ in angles:
        line, = ax.plot([], [], 'r-', alpha=0.8)
        rot_lines.append(line)
    
    # Text for step number
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Animation function
    def animate(i):
        step = i * 5  # 5 steps per frame
        
        for j, angle in enumerate(angles):
            # Original gradient
            grad_x, grad_y = math.cos(angle), math.sin(angle)
            
            # Previous gradient (assume orthogonal for visualization)
            prev_grad_x, prev_grad_y = -math.sin(angle), math.cos(angle)
            
            # Calculate pi_a and rotation angle
            cos_sim = grad_x * prev_grad_x + grad_y * prev_grad_y
            grad_angle = math.acos(max(-1.0, min(1.0, cos_sim)))
            pi_a = math.pi + 0.01 * math.tanh(grad_angle) + 0.01 * math.log1p(step)
            theta = pi_a / 16
            
            # Rotated gradient
            rot_x = grad_x * math.cos(theta) - prev_grad_x * math.sin(theta)
            rot_y = grad_y * math.cos(theta) - prev_grad_y * math.sin(theta)
            
            # Normalize
            norm = math.sqrt(rot_x**2 + rot_y**2)
            rot_x, rot_y = rot_x/norm, rot_y/norm
            
            rot_lines[j].set_data([0, rot_x], [0, rot_y])
        
        step_text.set_text(f'Step: {step}')
        
        return rot_lines + [step_text]
    
    ani = FuncAnimation(fig, animate, frames=100, interval=100, blit=True)
    ani.save(os.path.join(save_dir, 'pi_adaptive_animation.gif'), writer='pillow', fps=10)
    print(f"Animation saved to {save_dir}/pi_adaptive_animation.gif")
    
    plt.close('all')
    
    # Create visualization of how πₐ affects curved optimization paths
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define a simple 2D parabolic function
    def f(x, y):
        return x**2 + 5*y**2  # Elongated parabola
    
    # Create a contour plot
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    contour = ax.contour(X, Y, Z, 20, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    
    # Define starting point
    start_x, start_y = 1.8, 0.5
    
    # Simulate optimization paths for different optimizers
    paths = {
        'Gradient Descent': [],
        'GD with Momentum': [],
        'ARPPiA': []
    }
    
    current_positions = {
        'Gradient Descent': (start_x, start_y),
        'GD with Momentum': (start_x, start_y),
        'ARPPiA': (start_x, start_y)
    }
    
    # Previous gradients and momentum for respective methods
    prev_grad = {'GD with Momentum': (0, 0), 'ARPPiA': (0, 0)}
    momentum_buffer = {'GD with Momentum': (0, 0)}
    
    # Hyperparameters
    lr = 0.1
    momentum = 0.9
    alpha = 0.1
    mu = 0.01
    
    for step in range(40):
        for method in paths.keys():
            x, y = current_positions[method]
            paths[method].append((x, y))
            
            # Calculate gradient
            grad_x = 2 * x
            grad_y = 10 * y  # Stronger gradient in y direction
            
            if method == 'Gradient Descent':
                # Simple gradient descent update
                next_x = x - lr * grad_x
                next_y = y - lr * grad_y
                
            elif method == 'GD with Momentum':
                # Gradient descent with momentum
                m_x, m_y = momentum_buffer[method]
                m_x = momentum * m_x + grad_x
                m_y = momentum * m_y + grad_y
                
                next_x = x - lr * m_x
                next_y = y - lr * m_y
                
                momentum_buffer[method] = (m_x, m_y)
                
            elif method == 'ARPPiA':
                # ARPPiA update
                prev_grad_x, prev_grad_y = prev_grad[method]
                
                # Calculate angle between current and previous gradient
                grad_dot_prev = grad_x * prev_grad_x + grad_y * prev_grad_y
                grad_norm = math.sqrt(grad_x**2 + grad_y**2)
                prev_norm = math.sqrt(prev_grad_x**2 + prev_grad_y**2) if prev_norm != 0 else 1.0
                
                cos_sim = grad_dot_prev / (grad_norm * prev_norm) if prev_norm != 0 else 0
                cos_sim = max(-1.0, min(1.0, cos_sim))  # Clamp to valid range
                
                angle = math.acos(cos_sim) if abs(cos_sim) < 1.0 else 0
                pi_a = math.pi + 0.01 * math.tanh(angle) + 0.01 * math.log1p(step)
                theta = pi_a / 16
                
                # Rotate gradient
                rot_grad_x = grad_x * math.cos(theta) - prev_grad_x * math.sin(theta)
                rot_grad_y = grad_y * math.cos(theta) - prev_grad_y * math.sin(theta)
                
                # Update
                next_x = x - lr * rot_grad_x
                next_y = y - lr * rot_grad_y
                
                prev_grad[method] = (rot_grad_x, rot_grad_y)
            
            current_positions[method] = (next_x, next_y)
    
    # Plot optimization paths
    for method, path in paths.items():
        xs, ys = zip(*path)
        ax.plot(xs, ys, 'o-', markersize=4, label=method)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Optimization Paths on Curved Landscape')
    ax.legend()
    
    plt.savefig(os.path.join(save_dir, 'curved_optimization_paths.png'))
    print(f"Curved optimization paths visualization saved to {save_dir}/curved_optimization_paths.png")
    
    plt.close('all')

if __name__ == "__main__":
    visualize_pi_adaptive_rotation()
