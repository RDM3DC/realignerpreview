# ML Benchmarks for RealignR

This folder contains reference scripts for evaluating the RealignR optimizer on
vision and language tasks.

## CIFAR-10 / ResNet-18

`cifar10_resnet18.py` trains a standard ResNet-18 for 200 epochs with cosine
learning rate. Five optimization tracks are available:

1. **AdamW** – baseline.
2. **AdaGrad** – baseline with per-coordinate adaptive steps.
3. **AdamW→AdaGrad** – switch to AdaGrad when a plateau is detected.
4. **AdamW→RealignR** – switch to RealignR when a plateau is detected.
5. **AdamW→RealignR+CMA** – RealignR with curvature-modulated gain (``cma_xi=0.1``
   and ``cma_beta=0.05``).

The script logs epoch metrics to `logs/cifar10_resnet18_<track>.csv` including
training/validation loss, accuracy, gradient variance and signal-to-noise ratio.

## Tiny Transformer / WikiText-103

`tiny_transformer_wikitext.py` trains a GPT2-small style model (12 layers, 12
heads, width 768) on a subset of WikiText-103 for 100k steps. The same five
optimization tracks as above are provided. Validation perplexity and gradient
statistics are logged every 1k steps to
`logs/tiny_transformer_<track>.csv`.

Both scripts use `utils/plateau.py` for plateau detection. They are designed for
reproducible benchmarks rather than maximally efficient training.
