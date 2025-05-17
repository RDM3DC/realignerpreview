"""
ARPPiA Rotation Impact Visualizer

This script creates visualizations to understand the impact of
Pi-Adaptive rotation on gradient updates during neural network training.
It shows how the rotation mechanism affects the update direction
and magnitude compared to standard optimization approaches.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

def visualize_rotation_impact(save_dir='results/rotation_impact'):
    """Create visualizations to illustrate how πₐ rotation affects gradient updates."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create custom colormap for better visualization
    colors = [(0.6, 0, 0), (1, 0, 0), (1, 0.8, 0), (0, 1, 0), (0, 0.8, 1), (0, 0, 1), (0.6, 0, 0.6)]
    cmap_name = 'angle_colormap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    # 1. Visualize how rotation angle changes with gradient angle and training step
    fig = plt.figure(figsize=(16, 12))
    
    # Grid of angles and steps
    angles = np.linspace(0, np.pi, 100)
    steps = np.linspace(0, 5000, 100)
    
    angle_grid, step_grid = np.meshgrid(angles, steps)
    rotation_angles = np.zeros_like(angle_grid)
    
    # Calculate rotation angles
    for i in range(angle_grid.shape[0]):
        for j in range(angle_grid.shape[1]):
            angle = angle_grid[i, j]
            step = step_grid[i, j]
            
            # Original ARPPiA formula
            pi_a = np.pi + 0.01 * np.tanh(angle) + 0.01 * np.log1p(step)
            rotation_angles[i, j] = pi_a / 16  # Fixed divisor in original
    
    # Plot rotation angle as function of gradient angle and training step
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(rotation_angles, extent=[0, np.pi, 0, 5000], 
                    aspect='auto', origin='lower', cmap=cm)
    plt.colorbar(im1, ax=ax1, label='Rotation Angle (rad)')
    ax1.set_title('Original ARPPiA Rotation Angle')
    ax1.set_xlabel('Gradient Angle θ (rad)')
    ax1.set_ylabel('Training Step')
    
    # Calculate enhanced version with adaptive rotation
    enhanced_rotation_angles = np.zeros_like(angle_grid)
    for i in range(angle_grid.shape[0]):
        for j in range(angle_grid.shape[1]):
            angle = angle_grid[i, j]
            step = step_grid[i, j]
            
            # Enhanced ARPPiA formula with adaptive divisor
            pi_a = np.pi + 0.01 * np.tanh(angle) + 0.01 * np.log1p(step)
            divisor = 16 + 0.1 * np.log1p(step)  # Adaptive divisor
            enhanced_rotation_angles[i, j] = pi_a / divisor
    
    # Plot enhanced rotation angle
    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(enhanced_rotation_angles, extent=[0, np.pi, 0, 5000], 
                     aspect='auto', origin='lower', cmap=cm)
    plt.colorbar(im2, ax=ax2, label='Rotation Angle (rad)')
    ax2.set_title('ARPPiA+ Rotation Angle (Adaptive Divisor)')
    ax2.set_xlabel('Gradient Angle θ (rad)')
    ax2.set_ylabel('Training Step')
    
    # 2. Compare rotation angles between versions
    ax3 = fig.add_subplot(223)
    
    # Plot for fixed step = 1000
    step_idx = 20  # Approximately step 1000
    ax3.plot(angles, rotation_angles[step_idx], label='Original ARPPiA')
    ax3.plot(angles, enhanced_rotation_angles[step_idx], label='ARPPiA+')
    ax3.set_title(f'Rotation Angle vs. Gradient Angle (Step ≈ 1000)')
    ax3.set_xlabel('Gradient Angle θ (rad)')
    ax3.set_ylabel('Rotation Angle (rad)')
    ax3.legend()
    ax3.grid(True)
    
    # 3. Compare rotation evolution over training
    ax4 = fig.add_subplot(224)
    
    # Plot for fixed angle = π/2
    angle_idx = 50  # Approximately π/2
    ax4.plot(steps, rotation_angles[:, angle_idx], label='Original ARPPiA')
    ax4.plot(steps, enhanced_rotation_angles[:, angle_idx], label='ARPPiA+')
    ax4.set_title(f'Rotation Angle vs. Training Step (Angle ≈ π/2)')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Rotation Angle (rad)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rotation_angle_analysis.png'))
    print(f"Rotation angle analysis saved to {save_dir}/rotation_angle_analysis.png")
    
    # 4. Create 3D visualization of gradient updates
    fig = plt.figure(figsize=(18, 10))
    
    # Original gradient direction
    original_grad = np.array([1.0, 0.0])  # Unit vector along x-axis
    
    # Previous gradient (different angles)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    prev_grads = [np.array([np.cos(angle), np.sin(angle)]) for angle in angles]
    
    # Training steps to visualize
    steps = [0, 100, 1000, 5000]
    
    for i, step in enumerate(steps):
        # Plot for original ARPPiA
        ax1 = fig.add_subplot(2, 4, i+1, projection='3d')
        
        # Draw original gradient
        ax1.quiver(0, 0, 0, original_grad[0], original_grad[1], 0, 
                  color='blue', arrow_length_ratio=0.1, label='Original Gradient')
        
        # Draw rotated gradients for different previous gradients
        for prev_grad in prev_grads:
            # Calculate angle between gradients
            cos_sim = np.dot(original_grad, prev_grad)
            cos_sim = max(-1.0, min(1.0, cos_sim))
            angle = np.arccos(cos_sim)
            
            # Calculate πₐ and rotation angle
            pi_a = np.pi + 0.01 * np.tanh(angle) + 0.01 * np.log1p(step)
            theta = pi_a / 16
            
            # Rotate gradient
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            rotated_grad = np.dot(rotation_matrix, original_grad - prev_grad * np.sin(theta))
            
            # Normalize
            rotated_grad = rotated_grad / np.linalg.norm(rotated_grad)
            
            # Draw rotated gradient
            ax1.quiver(0, 0, 0, rotated_grad[0], rotated_grad[1], 0, 
                      color='red', arrow_length_ratio=0.1, alpha=0.5)
            
            # Draw previous gradient
            ax1.quiver(0, 0, 0, prev_grad[0], prev_grad[1], 0, 
                      color='green', arrow_length_ratio=0.1, alpha=0.3)
        
        ax1.set_title(f'Original ARPPiA (Step {step})')
        ax1.set_xlim([-1.2, 1.2])
        ax1.set_ylim([-1.2, 1.2])
        ax1.set_zlim([-0.5, 0.5])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # Plot for ARPPiA+
        ax2 = fig.add_subplot(2, 4, i+5, projection='3d')
        
        # Draw original gradient
        ax2.quiver(0, 0, 0, original_grad[0], original_grad[1], 0, 
                  color='blue', arrow_length_ratio=0.1, label='Original Gradient')
        
        # Draw rotated gradients for different previous gradients
        for prev_grad in prev_grads:
            # Calculate angle between gradients
            cos_sim = np.dot(original_grad, prev_grad)
            cos_sim = max(-1.0, min(1.0, cos_sim))
            angle = np.arccos(cos_sim)
            
            # Calculate πₐ and adaptive rotation angle
            pi_a = np.pi + 0.01 * np.tanh(angle) + 0.01 * np.log1p(step)
            divisor = 16 + 0.1 * np.log1p(step)
            theta = pi_a / divisor
            
            # Rotate gradient
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            rotated_grad = np.dot(rotation_matrix, original_grad - prev_grad * np.sin(theta))
            
            # Normalize
            rotated_grad = rotated_grad / np.linalg.norm(rotated_grad)
            
            # Draw rotated gradient
            ax2.quiver(0, 0, 0, rotated_grad[0], rotated_grad[1], 0, 
                      color='red', arrow_length_ratio=0.1, alpha=0.5)
            
            # Draw previous gradient
            ax2.quiver(0, 0, 0, prev_grad[0], prev_grad[1], 0, 
                      color='green', arrow_length_ratio=0.1, alpha=0.3)
        
        ax2.set_title(f'ARPPiA+ (Step {step})')
        ax2.set_xlim([-1.2, 1.2])
        ax2.set_ylim([-1.2, 1.2])
        ax2.set_zlim([-0.5, 0.5])
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradient_rotation_3d.png'))
    print(f"Gradient rotation 3D visualization saved to {save_dir}/gradient_rotation_3d.png")
    
    # 5. Create animation of rotation evolution during training
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title('Evolution of Gradient Rotation During Training')
    
    # Original gradient (fixed)
    orig_grad = np.array([1.0, 0.0])
    orig_arrow = ax.arrow(0, 0, orig_grad[0], orig_grad[1], 
                         head_width=0.05, head_length=0.1, 
                         fc='blue', ec='blue', label='Original Gradient')
    
    # Previous gradient (fixed perpendicular for clarity)
    prev_grad = np.array([0.0, 1.0])
    prev_arrow = ax.arrow(0, 0, prev_grad[0], prev_grad[1], 
                         head_width=0.05, head_length=0.1, 
                         fc='green', ec='green', alpha=0.5, label='Previous Gradient')
    
    # Create arrows for rotated gradients
    orig_rot_arrow = ax.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.1, 
                             fc='red', ec='red', label='Original ARPPiA')
    enhanced_rot_arrow = ax.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.1, 
                                 fc='purple', ec='purple', label='ARPPiA+')
    
    # Text for step number
    step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    
    ax.legend()
    
    def update(frame):
        step = frame * 100  # Multiply by 100 to show steps 0, 100, 200, etc.
        
        # Calculate πₐ values and rotation angles
        angle = np.pi/2  # Fixed 90-degree angle between gradients
        
        # Original ARPPiA
        pi_a_orig = np.pi + 0.01 * np.tanh(angle) + 0.01 * np.log1p(step)
        theta_orig = pi_a_orig / 16
        
        # Enhanced ARPPiA with adaptive divisor
        pi_a_enhanced = np.pi + 0.01 * np.tanh(angle) + 0.01 * np.log1p(step)
        divisor = 16 + 0.1 * np.log1p(step)
        theta_enhanced = pi_a_enhanced / divisor
        
        # Calculate rotated gradients
        rot_matrix_orig = np.array([
            [np.cos(theta_orig), -np.sin(theta_orig)],
            [np.sin(theta_orig), np.cos(theta_orig)]
        ])
        rotated_grad_orig = np.dot(rot_matrix_orig, orig_grad - prev_grad * np.sin(theta_orig))
        rotated_grad_orig = rotated_grad_orig / np.linalg.norm(rotated_grad_orig)
        
        rot_matrix_enhanced = np.array([
            [np.cos(theta_enhanced), -np.sin(theta_enhanced)],
            [np.sin(theta_enhanced), np.cos(theta_enhanced)]
        ])
        rotated_grad_enhanced = np.dot(rot_matrix_enhanced, orig_grad - prev_grad * np.sin(theta_enhanced))
        rotated_grad_enhanced = rotated_grad_enhanced / np.linalg.norm(rotated_grad_enhanced)
        
        # Update arrows
        orig_rot_arrow.set_data(x=0, y=0, dx=rotated_grad_orig[0], dy=rotated_grad_orig[1])
        enhanced_rot_arrow.set_data(x=0, y=0, dx=rotated_grad_enhanced[0], dy=rotated_grad_enhanced[1])
        
        # Update step text
        step_text.set_text(f'Step: {step}')
        
        return orig_rot_arrow, enhanced_rot_arrow, step_text
    
    anim = FuncAnimation(fig, update, frames=50, interval=200, blit=True)
    anim.save(os.path.join(save_dir, 'rotation_evolution.gif'), writer='pillow', fps=5)
    plt.close(fig)
    print(f"Rotation evolution animation saved to {save_dir}/rotation_evolution.gif")
    
    # 6. Compare with SGD + Momentum on a simple quadratic function
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define a simple quadratic function f(x,y) = x^2 + 5y^2
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 5*Y**2
    
    # Plot contours
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    
    # Define starting point
    start_x, start_y = 1.8, 0.5
    
    # Simulate optimization paths
    methods = {
        'Gradient Descent': {'color': 'blue', 'path': [(start_x, start_y)]},
        'GD + Momentum': {'color': 'green', 'path': [(start_x, start_y)]},
        'ARPPiA': {'color': 'red', 'path': [(start_x, start_y)]},
        'ARPPiA+': {'color': 'purple', 'path': [(start_x, start_y)]}
    }
    
    # Parameters
    lr = 0.1
    momentum = 0.9
    alpha = 0.1
    mu = 0.01
    
    # State variables
    momentum_buffer = {'GD + Momentum': (0, 0)}
    prev_grad = {'ARPPiA': (0, 0), 'ARPPiA+': (0, 0)}
    conductance = {'ARPPiA': 1.0, 'ARPPiA+': 1.0}
    
    # Run optimization
    for step in range(40):
        for method in methods:
            x, y = methods[method]['path'][-1]
            
            # Calculate gradient
            grad_x = 2 * x
            grad_y = 10 * y
            
            if method == 'Gradient Descent':
                # Simple gradient descent
                new_x = x - lr * grad_x
                new_y = y - lr * grad_y
                
            elif method == 'GD + Momentum':
                # Momentum update
                m_x, m_y = momentum_buffer[method]
                m_x = momentum * m_x + grad_x
                m_y = momentum * m_y + grad_y
                
                new_x = x - lr * m_x
                new_y = y - lr * m_y
                
                momentum_buffer[method] = (m_x, m_y)
                
            elif method == 'ARPPiA':
                # ARPPiA update
                prev_grad_x, prev_grad_y = prev_grad[method]
                
                # Calculate angle between gradients
                grad_norm = np.sqrt(grad_x**2 + grad_y**2)
                prev_norm = np.sqrt(prev_grad_x**2 + prev_grad_y**2) if (prev_grad_x**2 + prev_grad_y**2) > 0 else 1.0
                
                cos_sim = ((grad_x * prev_grad_x + grad_y * prev_grad_y) / 
                          (grad_norm * prev_norm)) if prev_norm > 0 else 0
                cos_sim = max(-1.0, min(1.0, cos_sim))
                
                angle = np.arccos(cos_sim) if abs(cos_sim) < 1.0 else 0
                pi_a = np.pi + 0.01 * np.tanh(angle) + 0.01 * np.log1p(step)
                theta = pi_a / 16
                
                # Rotate gradient
                rot_x = grad_x * np.cos(theta) - prev_grad_x * np.sin(theta)
                rot_y = grad_y * np.cos(theta) - prev_grad_y * np.sin(theta)
                
                # Update conductance
                conductance[method] = conductance[method] + alpha * np.sqrt(rot_x**2 + rot_y**2) - mu * conductance[method]
                
                # Apply update
                new_x = x - lr * conductance[method] * rot_x
                new_y = y - lr * conductance[method] * rot_y
                
                prev_grad[method] = (grad_x, grad_y)
                
            elif method == 'ARPPiA+':
                # ARPPiA+ update
                prev_grad_x, prev_grad_y = prev_grad[method]
                
                # Calculate angle between gradients
                grad_norm = np.sqrt(grad_x**2 + grad_y**2)
                prev_norm = np.sqrt(prev_grad_x**2 + prev_grad_y**2) if (prev_grad_x**2 + prev_grad_y**2) > 0 else 1.0
                
                cos_sim = ((grad_x * prev_grad_x + grad_y * prev_grad_y) / 
                          (grad_norm * prev_norm)) if prev_norm > 0 else 0
                cos_sim = max(-1.0, min(1.0, cos_sim))
                
                angle = np.arccos(cos_sim) if abs(cos_sim) < 1.0 else 0
                pi_a = np.pi + 0.01 * np.tanh(angle) + 0.01 * np.log1p(step)
                divisor = 16 + 0.1 * np.log1p(step)
                theta = pi_a / divisor
                
                # Rotate gradient with warmup
                warmup = min(1.0, step / 10)  # 10-step warmup
                rot_x = grad_x * np.cos(theta) - prev_grad_x * np.sin(theta) * warmup
                rot_y = grad_y * np.cos(theta) - prev_grad_y * np.sin(theta) * warmup
                
                # Update conductance with warmup
                conductance[method] = (conductance[method] + 
                                      alpha * warmup * np.sqrt(rot_x**2 + rot_y**2) - 
                                      mu * conductance[method])
                
                # Apply update
                new_x = x - lr * conductance[method] * rot_x
                new_y = y - lr * conductance[method] * rot_y
                
                prev_grad[method] = (grad_x, grad_y)
            
            methods[method]['path'].append((new_x, new_y))
    
    # Plot optimization paths
    for method, data in methods.items():
        path = data['path']
        xs, ys = zip(*path)
        ax.plot(xs, ys, 'o-', color=data['color'], label=method, linewidth=2, markersize=4)
    
    # Mark starting point
    ax.plot(start_x, start_y, 'ko', markersize=8)
    
    ax.set_title('Optimization Paths Comparison')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimization_paths_comparison.png'))
    print(f"Optimization paths comparison saved to {save_dir}/optimization_paths_comparison.png")

if __name__ == "__main__":
    visualize_rotation_impact()
