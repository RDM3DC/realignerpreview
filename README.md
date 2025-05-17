# ARPPiAGradientDescent Optimizer Suite

## Overview

ARPPiAGradientDescent is a novel optimization algorithm that combines Activity-Regulated Plasticity (ARP) with Pi-Adaptive (πₐ) rotation for neural network training. This repository contains implementations, enhanced versions, and comprehensive experiments showcasing the optimizer's performance and behavior.

## Key Features

- **Activity-Regulated Plasticity (ARP)**: Inspired by neurobiological principles, this approach dynamically adapts learning rates based on parameter-specific activity patterns.
  
- **Pi-Adaptive (πₐ) Rotation**: Improves convergence by adaptively rotating gradients based on the angular relationship between consecutive updates.

- **Adaptive Learning**: Each parameter gets its own conductance (G) value that evolves during training, enabling fine-grained control over the learning process.

## Optimizer Versions

### ARPPiAGradientDescent (Original)
The original implementation of the ARP with Pi-Adaptive rotation approach, featuring:
- Parameter-specific conductance (G) values
- Pi-Adaptive (πₐ) angle calculation for gradient rotation
- Gradient-activity based conductance updates

### ARPPiAGradientDescentPlus (Enhanced)
An enhanced version with additional features:
- **Adaptive Rotation Factor**: Dynamically adjusts the rotation amount based on training stage
- **Warmup Period**: Gradual increase in conductance activity to prevent early instability
- **Momentum Integration**: Combines rotation with traditional momentum for smoother updates
- **Adaptive Decay**: Adjusts decay coefficients based on gradient statistics

## Analysis Tools & Experiments

### Leaderboard & Dashboard System

The repository includes a comprehensive leaderboard and dashboard system:

- **Automated Metric Collection**: Scans result directories to compile metrics from all runs
- **Performance Comparison**: Ranks and compares runs by key metrics like loss, accuracy, and conductance
- **Interactive Dashboard**: Generates visual dashboards with plots and tables
- **Dataset Switching Analysis**: Special tools to analyze performance during dataset transitions

### Prerequisites

- Python 3.6 or higher
- PyTorch 1.8 or higher
- Matplotlib for visualization
- NumPy
- Pandas for data processing
- Seaborn for advanced plotting
- torchvision (for MNIST examples)

### Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/yourusername/ARPPiAGradientDescent.git
cd ARPPiAGradientDescent
pip install -r requirements.txt
```

### Basic Usage

```python
from arp_pia_optimizer import ARPPiAGradientDescent

# Initialize your model
model = YourModel()

# Create optimizer
optimizer = ARPPiAGradientDescent(
    model.parameters(),
    lr=0.01,     # Base learning rate
    alpha=0.01,  # Activity coefficient
    mu=0.001     # Decay coefficient
)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        # Forward pass
        output = model(input)
        loss = loss_function(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Examples

### MNIST Classification

Run the MNIST example:

```bash
python mnist_with_arp_pia.py
```

This will train a simple neural network on MNIST using ARPPiAGradientDescent and compare it with SGD. Results will be saved to the `results/mnist` directory.

### Visualizations

To visualize the behavior of the optimizer:

```bash
python visualize_arp_results.py
```

This creates:
- Evolution of G and πₐ values during training
- Phase portraits showing how these parameters interact

### Leaderboard System

Generate a leaderboard from your experimental results:

```bash
python leaderboard_autoloader.py --dirs results/run1 results/run2 --metric val_loss
```

This scans directories for metrics.csv files, extracts key information, and generates:
- A CSV file with all run metrics (leaderboard.csv)
- A JSON file with detailed information (leaderboard.json)
- A Markdown table for quick viewing (leaderboard.md)

### Dashboard Generation

Create visual dashboards from the leaderboard data:

```bash
python leaderboard_dashboard.py --input leaderboard.csv --output-dir dashboard
```

This generates:
- Performance comparison charts across runs
- Optimization parameter visualizations
- Dataset-model performance matrices
- RealignR advantage metrics

### Dataset Switching Analysis

Analyze how well models adapt when switching between datasets:

```bash
python analyze_dataset_switching.py
```

This finds runs with dataset switching, computes recovery rates, and creates visualizations showing adaptation performance.
- Animated visualizations of the optimization process

## How It Works

### ARP Component

The conductance G for each parameter is updated according to:

```
G ← G + α|I| - μG
```

Where:
- G is the conductance matrix
- α is the activity coefficient
- |I| is the absolute value of the rotated gradient
- μ is the decay coefficient

### πₐ Component

The Pi-Adaptive value is calculated as:

```
πₐ = π + c₁ * tanh(θ) + c₂ * log(1 + step)
```

Where:
- θ is the angle between current and previous gradients
- c₁ and c₂ are small constants (typically 0.01)
- step is the current optimization step

This value determines the rotation angle for the gradient.

## Results

On the MNIST dataset, ARPPiAGradientDescent shows:
- Fast initial convergence
- Stable learning dynamics
- Competitive final accuracy compared to standard optimizers

## Documentation

For more detailed information about the theory and implementation:
- See the [detailed documentation](ARPPiAGradientDescent_Documentation.md)
- Review the code comments in `arp_pia_optimizer.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work is inspired by research in neuroplasticity and adaptive learning systems
- Thanks to the PyTorch team for providing the foundation for implementing custom optimizers
