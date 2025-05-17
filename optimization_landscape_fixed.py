"""
Optimization Landscape Analysis for ARPPiAGradientDescent

This script analyzes the optimization trajectory and landscape
for different optimizers on a simple 2D test function.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import math
from mpl_toolkits.mplot3d import Axes3D
from arp_pia_optimizer import ARPPiAGradientDescent
from arp_pia_optimizer_plus import ARPPiAGradientDescentPlus

# Define 2D test functions

def rosenbrock(x, y, a=1, b=100):
    """Rosenbrock function - has a narrow valley with minimum at (a, a²)"""
    return (a - x)**2 + b * (y - x**2)**2

def himmelblau(x, y):
    """Himmelblau function - has 4 local minima"""
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def beale(x, y):
    """Beale function - has many steep regions and one global minimum"""
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def rastrigin(x, y, A=10):
    """Rastrigin function - highly multimodal with many local minima"""
    return 2*A + (x**2 - A*np.cos(2*np.pi*x)) + (y**2 - A*np.cos(2*np.pi*y))

# Wrapper classes for the test functions to make them optimizable with PyTorch

class TestFunction(torch.nn.Module):
    def __init__(self, func, x0, y0):
        super(TestFunction, self).__init__()
        self.func = func
        self.xy = torch.nn.Parameter(torch.tensor([x0, y0], dtype=torch.float32))
        
    def forward(self):
        x, y = self.xy
        return self.func(x, y)

# Function to visualize the optimization landscape and trajectory

def visualize_optimization(func, xlim, ylim, trajectories, title, save_path):
    # Create grid for contour plot
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Calculate function values
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(X[i, j], Y[i, j])
    
    # Create 3D and 2D plots
    fig = plt.figure(figsize=(18, 8))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    # Add trajectories to 3D plot
    for name, traj in trajectories.items():
        # Filter out nan values
        filtered_traj = [p for p in traj if not (math.isnan(p[0]) or math.isnan(p[1]))]
        if filtered_traj:
            xs, ys = zip(*filtered_traj)
            zs = [func(x, y) for x, y in filtered_traj]
            ax1.plot(xs, ys, zs, 'o-', markersize=3, linewidth=2, label=name)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X, Y)')
    ax1.set_title(f'3D Surface: {title}')
    
    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour, ax=ax2)
    
    # Add trajectories to contour plot
    for name, traj in trajectories.items():
        # Filter out nan values
        filtered_traj = [p for p in traj if not (math.isnan(p[0]) or math.isnan(p[1]))]
        if filtered_traj:
            xs, ys = zip(*filtered_traj)
            ax2.plot(xs, ys, 'o-', markersize=4, linewidth=2, label=name)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Contour: {title}')
    
    # Add legend
    ax2.legend()
    
    # Save and show
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    
    # Check if we have valid trajectories for animation
    valid_animation = False
    for traj in trajectories.values():
        if any(not (math.isnan(p[0]) or math.isnan(p[1])) for p in traj):
            valid_animation = True
            break
    
    if valid_animation:
        try:
            # Create animation of optimization path
            fig, ax = plt.figure(figsize=(10, 8)), plt.axes()
            contour = ax.contour(X, Y, Z, 50, cmap='viridis')
            plt.colorbar(contour)
            
            # Prepare lines and points for each trajectory
            lines = {}
            points = {}
            
            # Filter trajectories for animation
            filtered_trajectories = {}
            for name, traj in trajectories.items():
                filtered_traj = [p for p in traj if not (math.isnan(p[0]) or math.isnan(p[1]))]
                if filtered_traj:
                    filtered_trajectories[name] = filtered_traj
                    line, = ax.plot([], [], '-', linewidth=2, label=name)
                    point, = ax.plot([], [], 'o', markersize=6)
                    lines[name] = line
                    points[name] = point
            
            if not filtered_trajectories:
                print("No valid trajectories for animation")
                return
                
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(f'Optimization Trajectory: {title}')
            ax.legend()
            
            max_steps = max(len(traj) for traj in filtered_trajectories.values())
            
            def init():
                for name in filtered_trajectories:
                    lines[name].set_data([], [])
                    points[name].set_data([], [])
                return list(lines.values()) + list(points.values())
            
            def animate(i):
                for name, traj in filtered_trajectories.items():
                    if i < len(traj):
                        xs, ys = zip(*traj[:i+1])
                        lines[name].set_data(xs, ys)
                        points[name].set_data([traj[i][0]], [traj[i][1]])
                return list(lines.values()) + list(points.values())
            
            anim = FuncAnimation(fig, animate, frames=min(max_steps, 100),
                                init_func=init, blit=True, interval=100)
            
            # Save animation
            animation_path = save_path.replace('.png', '_animation.gif')
            anim.save(animation_path, writer='pillow', fps=10)
            print(f"Animation saved to {animation_path}")
            
        except Exception as e:
            print(f"Error creating animation: {e}")
        
    plt.close('all')

# Function to run optimization with different optimizers

def optimize_function(func, x0, y0, optimizers, steps=100):
    trajectories = {}
    final_values = {}
    
    for name, (optimizer_class, kwargs) in optimizers.items():
        print(f"Optimizing with {name}...")
        model = TestFunction(func, x0, y0)
        optimizer = optimizer_class(model.parameters(), **kwargs)
        
        # Keep track of the optimization trajectory
        trajectory = []
        trajectory.append(model.xy.data.tolist())
        
        for step in range(steps):
            try:
                optimizer.zero_grad()
                loss = model()
                loss.backward()
                optimizer.step()
                
                # Record the current position
                trajectory.append(model.xy.data.tolist())
                
                if step % 10 == 0:
                    print(f"  Step {step}: xy = {model.xy.data.tolist()}, loss = {loss.item():.6f}")
                    
                # Break early if we get NaN values to avoid excessive NaN entries
                if math.isnan(loss.item()):
                    print(f"  NaN loss encountered at step {step}. Stopping early.")
                    break
                    
            except Exception as e:
                print(f"  Error at step {step}: {e}")
                break
        
        # Get final values safely
        try:
            final_pos = model.xy.data.tolist()
            final_loss = loss.item() if 'loss' in locals() else float('nan')
        except:
            final_pos = [float('nan'), float('nan')]
            final_loss = float('nan')
            
        trajectories[name] = trajectory
        final_values[name] = (final_pos, final_loss)
        
        print(f"  Final position: xy = {final_pos}, loss = {final_loss:.6f}")
    
    return trajectories, final_values

# Main function to test optimizers on different functions

def test_optimizers(save_dir='results/optimization_landscape'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Define optimizers to compare with lower learning rates for stability
    optimizers = {
        'SGD': (torch.optim.SGD, {'lr': 0.001, 'momentum': 0.9}),
        'ARPPiA': (ARPPiAGradientDescent, {'lr': 0.001, 'alpha': 0.01, 'mu': 0.001}),
        'ARPPiA+': (ARPPiAGradientDescentPlus, {'lr': 0.001, 'alpha': 0.01, 'mu': 0.001, 
                                             'momentum': 0.9, 'warmup_steps': 20})
    }
    
    # Test on Rosenbrock function
    print("\nTesting on Rosenbrock function...")
    trajectories, final_values = optimize_function(
        rosenbrock, -1.0, 1.0, optimizers, steps=200)
    visualize_optimization(
        rosenbrock, [-2, 2], [-1, 3], 
        trajectories, "Rosenbrock Function", 
        os.path.join(save_dir, "rosenbrock.png"))
    
    # Test on Himmelblau function
    print("\nTesting on Himmelblau function...")
    trajectories, final_values = optimize_function(
        himmelblau, -3.0, -3.0, optimizers, steps=200)
    visualize_optimization(
        himmelblau, [-6, 6], [-6, 6], 
        trajectories, "Himmelblau Function", 
        os.path.join(save_dir, "himmelblau.png"))
    
    # Test on Beale function
    print("\nTesting on Beale function...")
    trajectories, final_values = optimize_function(
        beale, 1.0, 1.0, optimizers, steps=200)
    visualize_optimization(
        beale, [-4, 4], [-4, 4], 
        trajectories, "Beale Function", 
        os.path.join(save_dir, "beale.png"))
    
    # Test on Rastrigin function
    print("\nTesting on Rastrigin function...")
    trajectories, final_values = optimize_function(
        rastrigin, 2.5, 2.5, optimizers, steps=200)
    visualize_optimization(
        rastrigin, [-5, 5], [-5, 5], 
        trajectories, "Rastrigin Function", 
        os.path.join(save_dir, "rastrigin.png"))
    
    # Print summary
    print("\nOptimization Summary:")
    print("-" * 80)
    print("Function | Optimizer | Final Position | Final Loss")
    print("-" * 80)
    
    for func_name in ["Rosenbrock", "Himmelblau", "Beale", "Rastrigin"]:
        for opt_name, (final_pos, final_loss) in final_values.items():
            x_val = final_pos[0] if not math.isnan(final_pos[0]) else float('nan')
            y_val = final_pos[1] if not math.isnan(final_pos[1]) else float('nan')
            loss_val = final_loss if not math.isnan(final_loss) else float('nan')
            
            x_str = f"{x_val:.4f}" if not math.isnan(x_val) else "NaN"
            y_str = f"{y_val:.4f}" if not math.isnan(y_val) else "NaN"
            loss_str = f"{loss_val:.6f}" if not math.isnan(loss_val) else "NaN"
            
            print(f"{func_name:10} | {opt_name:8} | ({x_str}, {y_str}) | {loss_str}")
    
    print("-" * 80)

if __name__ == "__main__":
    test_optimizers()
