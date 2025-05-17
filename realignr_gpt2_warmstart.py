# === realignr_gpt2_warmstart.py ===
"""
Full training script with:
- GPT-2 pretrained model loading
- AdamW warm-up (5k steps)
- ARP optimizer takeover
- Safe loss masking (PAD -> -100)
- Packed Wikitext-103
"""
import torch, os
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AdamW
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader

from optimizers.arp_optimizer import ARPOptimizer

# === Config ===
CTX_LEN...
