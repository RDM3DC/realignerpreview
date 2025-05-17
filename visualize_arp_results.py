"""
Visualize ARPPiAGradientDescent Results

This script loads and visualizes the results from the MNIST experiments
with the ARPPiAGradientDescent optimizer.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from matplotlib.animation import FuncAnimation

def create_arp_animation(g_values, pi_values, save_path='arp_animation.gif'):
    """Create an animation showing how G and πₐ evolve together"""
    
    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # We'll show the last 100 points in the animation window
    window_size = 100
    
    # Determine the maximum number of frames
    max_frames = min(len(g_values), len(pi_values))
    
    # Define x-axis values (steps)
    steps = np.arange(max_frames)
    
    # Maximum value for y-axis limits with a small margin
    max_g = max(g_values) * 1.1
    max_pi = max(pi_values) * 1.1
    
    # Setup empty line objects
    line1, = ax1.plot([], [], 'b-', linewidth=2)
    point1, = ax1.plot([], [], 'ro', markersize=8)
    
    line2, = ax2.plot([], [], 'g-', linewidth=2)
    point2, = ax2.plot([], [], 'ro', markersize=8)
    
    # Set up plot parameters
    ax1.set_xlim(0, window_size)
    ax1.set_ylim(0, max_g)
    ax1.set_title('Conductance (G) Evolution')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('G Value')
    ax1.grid(True)
    
    ax2.set_xlim(0, window_size)
    ax2.set_ylim(min(pi_values), max_pi)
    ax2.set_title('Pi-Adaptive (πₐ) Evolution')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('πₐ Value')
    ax2.grid(True)
    
    # Text objects for current values
    g_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    pi_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
    step_text = fig.text(0.5, 0.01, '', ha='center')
    
    # Animation function
    def update(frame):
        # Calculate window start and end
        if frame < window_size:
            start = 0
            end = min(window_size, frame + 1)
            window_x = steps[:end]
        else:
            start = frame - window_size + 1
            end = frame + 1
            window_x = np.arange(window_size)
        
        # Get window of data
        window_g = g_values[start:end]
        window_pi = pi_values[start:end]
        
        # Update line data
        line1.set_data(window_x, window_g)
        point1.set_data([window_x[-1]], [window_g[-1]])
        
        line2.set_data(window_x, window_pi)
        point2.set_data([window_x[-1]], [window_pi[-1]])
        
        # Update text
        g_text.set_text(f'G: {window_g[-1]:.4f}')
        pi_text.set_text(f'πₐ: {window_pi[-1]:.4f}')
        step_text.set_text(f'Step: {frame} / {max_frames-1}')
        
        # Adjust x-axis limits for scrolling effect
        if frame >= window_size:
            ax1.set_xlim(0, window_size)
            ax2.set_xlim(0, window_size)
        else:
            ax1.set_xlim(0, end)
            ax2.set_xlim(0, end)
        
        return line1, point1, line2, point2, g_text, pi_text, step_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=max_frames, 
                        blit=True, interval=50, repeat=True)
    
    # Save animation
    try:
        ani.save(save_path, writer='pillow', fps=30)
        print(f"Animation saved to {save_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
    
    plt.close()

def create_phase_portrait(g_values, pi_values, save_path='phase_portrait.png'):
    """Create a phase portrait showing the relationship between G and πₐ"""
    
    # Setup the figure
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot with color gradient based on step
    num_points = min(len(g_values), len(pi_values))
    plt.scatter(g_values[:num_points], pi_values[:num_points], 
                c=np.arange(num_points), cmap='viridis', 
                alpha=0.6, s=10)
    
    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Training Step')
    
    # Add some arrows to show the direction of evolution
    subsample = max(1, num_points // 20)  # Show ~20 arrows
    for i in range(subsample, num_points, subsample):
        plt.arrow(g_values[i-subsample], pi_values[i-subsample],
                  g_values[i] - g_values[i-subsample],
                  pi_values[i] - pi_values[i-subsample],
                  head_width=0.02, head_length=0.03,
                  fc='red', ec='red', alpha=0.6)
    
    plt.title('Phase Portrait: G vs πₐ Evolution')
    plt.xlabel('Conductance (G)')
    plt.ylabel('Pi-Adaptive (πₐ)')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(save_path)
    print(f"Phase portrait saved to {save_path}")
    plt.close()

def visualize_mnist_results(results_dir='results/mnist'):
    """Load and visualize MNIST experiment results"""
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find plot files
    plot_files = glob.glob(os.path.join(results_dir, '*.png'))
    if not plot_files:
        print(f"No plot files found in {results_dir}")
        return
    
    print(f"Found {len(plot_files)} plot files:")
    for plot_file in plot_files:
        print(f" - {os.path.basename(plot_file)}")
    
    # Create a function to simulate G and πₐ data for visualization
    # In a real implementation, you would load the actual data from saved files
    def generate_synthetic_data(n_steps=1000):
        steps = np.linspace(0, 10, n_steps)
        g_values = 1.0 + 0.5 * np.sin(steps * 0.5) + 0.2 * np.random.randn(n_steps) + steps * 0.05
        g_values = np.clip(g_values, 0.5, 10.0)
        
        pi_values = np.pi + 0.01 * np.tanh(steps) + 0.01 * np.log1p(steps)
        pi_values += 0.05 * np.random.randn(n_steps)
        
        return g_values, pi_values
    
    # Generate synthetic data for demonstration
    g_values, pi_values = generate_synthetic_data()
    
    # Create an animation of G and πₐ evolution
    create_arp_animation(g_values, pi_values, 
                          save_path=os.path.join(results_dir, 'arp_pia_evolution.gif'))
    
    # Create a phase portrait
    create_phase_portrait(g_values, pi_values,
                           save_path=os.path.join(results_dir, 'arp_pia_phase_portrait.png'))
    
    print("Visualization complete!")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = 'results/mnist'
    os.makedirs(results_dir, exist_ok=True)
    
    # Run visualization
    visualize_mnist_results(results_dir)
