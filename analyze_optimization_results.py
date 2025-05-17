"""
Convergence Analysis for ARPPiAGradientDescent Optimizers

This script analyzes the convergence characteristics of the original ARPPiAGradientDescent 
and the enhanced ARPPiAGradientDescentPlus optimizers, comparing them against standard
optimizers like SGD.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
import glob
from matplotlib.gridspec import GridSpec

def load_results(base_dir='results/comparison'):
    """Load results from the comparison directory."""
    results = {}
    
    # Check if results file exists (saved by compare_optimizers.py)
    results_file = os.path.join(base_dir, 'results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        # Try to load individual model files
        model_files = glob.glob(os.path.join(base_dir, 'mnist_*.pth'))
        for file in model_files:
            optimizer_name = os.path.basename(file).replace('mnist_', '').replace('.pth', '')
            if 'sgd' in optimizer_name.lower():
                name = 'SGD'
            elif 'plus' in optimizer_name.lower() or '_plus' in optimizer_name.lower():
                name = 'ARPPiA+'
            else:
                name = 'ARPPiA'
                
            # Load model stats if available
            stats_file = file.replace('.pth', '_stats.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    results[name] = json.load(f)
    
    return results

def plot_convergence(results, save_dir='results/analysis'):
    """Plot convergence metrics from the results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    for name, metrics in results.items():
        if 'losses' in metrics:
            ax1.plot(metrics['losses'], marker='o', label=name)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Accuracy curves
    ax2 = fig.add_subplot(gs[0, 1])
    for name, metrics in results.items():
        if 'accuracy' in metrics:
            ax2.plot(metrics['accuracy'], marker='o', label=name)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Loss change rate
    ax3 = fig.add_subplot(gs[0, 2])
    for name, metrics in results.items():
        if 'losses' in metrics:
            losses = metrics['losses']
            # Calculate relative change in loss
            loss_changes = [abs(losses[i] - losses[i-1])/max(losses[i-1], 1e-8) 
                           for i in range(1, len(losses))]
            ax3.plot(range(1, len(losses)), loss_changes, marker='o', label=name)
    ax3.set_title('Relative Loss Change')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Relative Change')
    ax3.legend()
    ax3.grid(True)
    ax3.set_yscale('log')
    
    # 4. G value evolution (for ARP optimizers)
    ax4 = fig.add_subplot(gs[1, 0])
    for name, metrics in results.items():
        if 'g_logs' in metrics and metrics['g_logs']:
            ax4.plot(metrics['g_logs'], label=name)
    ax4.set_title('Conductance (G) Evolution')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('G Value')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Pi-Adaptive value evolution
    ax5 = fig.add_subplot(gs[1, 1])
    for name, metrics in results.items():
        if 'pi_logs' in metrics and metrics['pi_logs']:
            ax5.plot(metrics['pi_logs'], label=name)
    ax5.set_title('Pi-Adaptive (πₐ) Evolution')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('πₐ Value')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Test accuracy comparison
    ax6 = fig.add_subplot(gs[1, 2])
    names = []
    accuracies = []
    for name, metrics in results.items():
        if 'test_accuracy' in metrics:
            names.append(name)
            accuracies.append(metrics['test_accuracy'])
    
    bars = ax6.bar(names, accuracies)
    ax6.set_title('Test Accuracy')
    ax6.set_ylabel('Accuracy (%)')
    ax6.grid(True, axis='y')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.2f}%", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_analysis.png'))
    print(f"Convergence analysis saved to {save_dir}/convergence_analysis.png")
    
def analyze_gradient_statistics(results, save_dir='results/analysis'):
    """Analyze gradient statistics for different optimizers."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure for G-values vs πₐ relationship
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot G vs πₐ for ARP optimizers
    for name, metrics in results.items():
        if 'g_logs' in metrics and 'pi_logs' in metrics and metrics['g_logs'] and metrics['pi_logs']:
            # Take min length of both series
            min_len = min(len(metrics['g_logs']), len(metrics['pi_logs']))
            g_values = metrics['g_logs'][:min_len]
            pi_values = metrics['pi_logs'][:min_len]
            
            # Create scatter plot
            scatter = ax.scatter(pi_values, g_values, alpha=0.5, label=name)
            
            # Add trend line
            z = np.polyfit(pi_values, g_values, 1)
            p = np.poly1d(z)
            ax.plot(sorted(pi_values), p(sorted(pi_values)), 
                    linestyle='--', color=scatter.get_facecolor()[0])
    
    ax.set_title('Relationship Between G-Values and πₐ')
    ax.set_xlabel('πₐ Value')
    ax.set_ylabel('G Value')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'g_pi_relationship.png'))
    print(f"G-πₐ relationship analysis saved to {save_dir}/g_pi_relationship.png")
    
    # Create convergence phase portrait
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, metrics in results.items():
        if 'losses' in metrics and len(metrics['losses']) > 1:
            losses = metrics['losses']
            # Calculate loss change rate
            loss_changes = [losses[i] - losses[i-1] for i in range(1, len(losses))]
            
            # Plot loss vs loss change rate
            ax.plot(losses[1:], loss_changes, 'o-', label=name)
    
    ax.set_title('Convergence Phase Portrait')
    ax.set_xlabel('Loss')
    ax.set_ylabel('Loss Change')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_phase_portrait.png'))
    print(f"Convergence phase portrait saved to {save_dir}/convergence_phase_portrait.png")
    
    # If we have step times, create efficiency analysis
    has_step_times = any('step_times' in metrics for metrics in results.values())
    if has_step_times:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = []
        times = []
        accuracies = []
        
        for name, metrics in results.items():
            if 'step_times' in metrics and 'test_accuracy' in metrics:
                names.append(name)
                times.append(metrics['step_times'] * 1000)  # Convert to ms
                accuracies.append(metrics['test_accuracy'])
        
        # Create bar plot for efficiency (accuracy / step time)
        efficiency = [acc / time for acc, time in zip(accuracies, times)]
        bars = ax.bar(names, efficiency)
        
        ax.set_title('Optimizer Efficiency (Accuracy per Computation Time)')
        ax.set_ylabel('Efficiency (Accuracy % / ms)')
        ax.grid(True, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f"{height:.2f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'optimizer_efficiency.png'))
        print(f"Optimizer efficiency analysis saved to {save_dir}/optimizer_efficiency.png")

def main():
    results = load_results()
    if not results:
        print("No results found. Run compare_optimizers.py first.")
        return
        
    print(f"Found results for {list(results.keys())}")
    
    plot_convergence(results)
    analyze_gradient_statistics(results)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
