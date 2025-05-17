# ------------------------------------------------------------
# realignr_dataparallel.py — Windows-compatible DataParallel version 
# ------------------------------------------------------------
#  • Windows-compatible DataParallel setup (not DDP)
#  • Utilizes all available GPUs even on Windows
#  • Meta-controller runs in the main thread
#  • Based on the original realignr_resume_crossdomain_fixed.py
# ------------------------------------------------------------

import os
import time
import glob
import math
import json
import argparse
from pathlib import Path
from itertools import chain
import threading
import queue
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp

# ── local utilities --------------------------------------------------
from optimizers.arp_optimizer import ARPOptimizer
from meta_controller import MetaController, CPRController
from action import ActionTracker

# ── CONSTANTS --------------------------------------------------------
CURVATURE_MIN_THRESHOLD = 0.15
DELTA = 0.01
EPSILON = 0.001
INITIAL_DELTA = 0.01  # Initial value for DELTA
INITIAL_EPSILON = 0.001  # Initial value for EPSILON
L_MAX = 10.0
BASE_DIR = Path(__file__).resolve().parent
STEP_START = 0
CKPT_DIR = BASE_DIR / "checkpoints"
RESUME_CKPT = None

MAX_STEPS = 300_000
CKPT_INTERVAL = 2_000
LOG_INTERVAL = 50
MAX_BACKUPS = 8
PLATEAU_PATIENCE = 500
PLATEAU_SLOPE_THRESHOLD = -0.001

SEQ_LEN = 1_024
CTX_START = 1_024
BATCH_SIZE_PER_GPU = 2  # Effective batch size will be this * num_gpus
CONTEXT_SCHEDULE = [
    (50_000, 2_048),   # Expand to 2k at step 50k
    (120_000, 4_096),  # Expand to 4k at step 120k
]
ALPHA, MU = 0.01, 0.001

GEN_PROMPT = "The meaning of life is"
GEN_LEN = 50

# ── MODEL DEFINITION -------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        
    def forward(self, x, mask):
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        self.last_out = x.detach()
        return x

class RealignRGPT2(nn.Module):
    def __init__(self, vocab, ctx_len):
        super().__init__()
        dim = 768
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, ctx_len, dim))
        self.blocks = nn.ModuleList(Block(dim, 12) for _ in range(4))
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

        g_init = torch.zeros(len(self.blocks), dim)
        self.register_buffer("G", g_init)
        c_init = torch.ones(len(self.blocks), dim)
        self.register_buffer("C", c_init)

    def expand_ctx(self, new_len):
        if new_len <= self.ctx_len: return
        old = self.pos_emb.data 
        new = torch.zeros(1, new_len, old.size(2), device=old.device)
        new[:, :self.ctx_len] = old
        nn.init.trunc_normal_(new[:, self.ctx_len:], std=0.02)
        self.pos_emb, self.ctx_len = nn.Parameter(new), new_len
        print(f"🔁 context window → {new_len}")
        
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T]
        mask = torch.triu(torch.ones(T, T, device=idx.device)*float('-inf'), 1)

        activ_mean = []
        for blk in self.blocks:
            x = blk(x, mask)
            activ_mean.append(x.abs().mean((0, 1)))

        return self.head(self.ln_f(x)), activ_mean

    @torch.no_grad()
    def generate(self, prompt, max_len=GEN_LEN, temp=1.0, top_k=0):
        self.eval()
        ids = tokenizer.encode(prompt, return_tensors="pt").to(self.head.weight.device)
        for _ in range(max_len):
            logits, _ = self(ids[:, -self.ctx_len:])
            logits = logits[:, -1, :]/temp
            if top_k:
                topv, topi = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('-inf'))
                logits.scatter_(1, topi, topv)
            next_tok = torch.multinomial(torch.softmax(logits, -1), 1)
            ids = torch.cat([ids, next_tok], -1)
        self.train()
        return tokenizer.decode(ids[0], skip_special_tokens=True)

# ── Helper function for loss slope calculation -----------------------
def calculate_slope(loss_window):
    if not loss_window or len(loss_window) < 2:
        return 0
    y = torch.tensor(loss_window, dtype=torch.float32)
    x = torch.arange(len(y), dtype=torch.float32)
    N = len(x)
    sum_xy = torch.sum(x * y); sum_x = torch.sum(x); sum_y = torch.sum(y); sum_x_sq = torch.sum(x**2)
    denominator = N * sum_x_sq - sum_x**2
    return (N * sum_xy - sum_x * sum_y) / denominator if denominator != 0 else 0

# Helper function for curvature variance checking
def curvature_variance_high(C):
    c_std = C.std().item()
    c_mean = C.mean().item()
    return c_mean > 0 and (c_std / c_mean) > 0.5

# ── Main Worker Function ---------------------------------------------
def main_worker(gpu_id, num_gpus, args):
    # Basic setup for this worker
    rank = gpu_id
    is_main_process = (rank == 0)
    use_multi_gpu = num_gpus > 1
    
    # Set device for this worker
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')  # Primary GPU for DataParallel
        print(f"CUDA available with {torch.cuda.device_count()} devices")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA is not available or no GPUs detected. Running on CPU.")
    
    print(f"Using device: {device}")

    # Create logging directories
    run_id = int(time.time())
    LOG_DIR = BASE_DIR / "runs" / f"dataparallel_{run_id}"
    if is_main_process:
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(CKPT_DIR, exist_ok=True)
        print(f"Log directory: {LOG_DIR}")
    
    # Initialize tokenizer
    global tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    def tokenize_and_chunk(ex):
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        flat = list(chain.from_iterable(ids))
        return {"input_ids": [flat[i:i+SEQ_LEN] for i in range(0, len(flat), SEQ_LEN)]}
    
    train_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:2%]")
    train_dataset = train_dataset.map(tokenize_and_chunk, batched=True, remove_columns=["text"])
    train_dataset = train_dataset.filter(lambda e: len(e["input_ids"]) == SEQ_LEN).with_format("torch")
    
    # Calculate batch size - for DataParallel, we use a smaller effective batch size
    # since it will be multiplied across GPUs
    batch_size = BATCH_SIZE_PER_GPU
    if use_multi_gpu and hasattr(args, 'use_data_parallel') and args.use_data_parallel:
        available_gpus = torch.cuda.device_count()
        print(f"Adjusting batch size for {available_gpus} GPUs")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = RealignRGPT2(tokenizer.vocab_size, ctx_len=CTX_START).to(device)
    
    # Wrap model with DataParallel if using multiple GPUs
    if use_multi_gpu and hasattr(args, 'use_data_parallel') and args.use_data_parallel:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        # Get all available GPUs
        device_ids = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # Create optimizer
    optimizer = ARPOptimizer(model.parameters(), alpha=ALPHA, mu=MU)
    
    # Check if optimizer has direct attributes or uses param_groups
    try:
        print(f"ARPOptimizer attributes - alpha: {optimizer.alpha}, mu: {optimizer.mu}")
        uses_direct_attributes = True
    except AttributeError:
        print("ARPOptimizer uses param_groups instead of direct attributes")
        uses_direct_attributes = False
        if len(optimizer.param_groups) > 0:
            if 'alpha' in optimizer.param_groups[0] and 'mu' in optimizer.param_groups[0]:
                print(f"ARPOptimizer param_groups - alpha: {optimizer.param_groups[0]['alpha']}, mu: {optimizer.param_groups[0]['mu']}")
    
    # Only create GradScaler if CUDA is available
    use_amp = torch.cuda.is_available()
    
    # Define a dummy context manager for when AMP isn't used
    class DummyContextManager:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def get_scale(self): return 1.0
    
    # Setup AMP components
    if use_amp:
        scaler = GradScaler()
        autocast_ctx = autocast()
    else:
        scaler = DummyContextManager()
        autocast_ctx = DummyContextManager()
        print("Running without AMP (using dummy scaler)")
    
    # Create TensorBoard writer
    writer = SummaryWriter(str(LOG_DIR), purge_step=STEP_START)
    print(f"[TensorBoard] Run: tensorboard --logdir \"{LOG_DIR}\"")
    
    # Create controllers
    cpr_controller = CPRController(epsilon=1e-3, reset_patience=500)
    meta_controller = MetaController()
    
    # Training loop variables
    step = STEP_START
    ema = None
    ctx_idx = 0
    loss_history = []
    last_curv_warn_step = -100
    
    # Get access to model.module if using DataParallel
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # Main training loop
    print(f"Starting training...")
    for epoch in range(MAX_STEPS // len(train_loader) + 1):
        for batch_idx, batch in enumerate(train_loader):
            if step >= MAX_STEPS:
                break
            
            # Move data to the correct device
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            
            # Handle hyperparameter adjustments
            # Curvature memory C logging
            writer.add_scalar('Curvature/C_mean', core.C.mean().item(), step)
            writer.add_scalar('Curvature/C_std', core.C.std().item(), step)
            if step % 1000 == 0:
                writer.add_histogram('Curvature/C_hist', core.C, step)
            
            # Dynamic curvature learning rate adjustment
            # Initialize DELTA and EPSILON from function attributes or constants
            DELTA = getattr(main_worker, "DELTA", INITIAL_DELTA)
            EPSILON = getattr(main_worker, "EPSILON", INITIAL_EPSILON)
            
            c_mean = core.C.mean().item()
            if c_mean < CURVATURE_MIN_THRESHOLD:
                DELTA = DELTA * 1.05
                main_worker.DELTA = DELTA  # Store in function attribute for persistence
                print(f"Increasing curvature learning rate δ to {DELTA:.4f} at step {step}")
            
            # Log DELTA and EPSILON
            writer.add_scalar('Curvature/DELTA', DELTA, step)
            writer.add_scalar('Curvature/EPSILON', EPSILON, step)
            
            # Check curvature variance
            if curvature_variance_high(core.C):
                EPSILON = EPSILON * 0.95
                main_worker.EPSILON = EPSILON  # Store in function attribute for persistence
                print(f"Reducing curvature decay ε to {EPSILON:.5f} at step {step}")
            
            # Dynamic context expansion
            if ctx_idx < len(CONTEXT_SCHEDULE) and step >= CONTEXT_SCHEDULE[ctx_idx][0]:
                target_ctx = CONTEXT_SCHEDULE[ctx_idx][1]
                if target_ctx > core.ctx_len:
                    print(f"🔁 Expanding context window: {core.ctx_len} → {target_ctx} at step {step}")
                    core.expand_ctx(target_ctx)
                else:
                    print(f"Skipping context schedule entry {CONTEXT_SCHEDULE[ctx_idx]} as target length is not greater than current {core.ctx_len}")
                ctx_idx += 1
            
            # Forward pass and loss calculation
            optimizer.zero_grad(set_to_none=True)
            
            with autocast_ctx:
                logits, activ_mean = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
                    labels[:, 1:].reshape(-1)
                )
            
            # Backward pass with AMP
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Get loss value
            current_loss_item = loss.item()
            
            # Update EMA and store loss history
            ema = current_loss_item if ema is None else 0.99*ema + 0.01*current_loss_item
            loss_history.append(current_loss_item)
            if len(loss_history) > PLATEAU_PATIENCE * 2:
                loss_history.pop(0)
            
            # G and C updates for ARPOptimizer
            if isinstance(optimizer, ARPOptimizer):
                # Ensure DELTA and EPSILON are defined for this process
                DELTA = getattr(main_worker, "DELTA", INITIAL_DELTA)
                EPSILON = getattr(main_worker, "EPSILON", INITIAL_EPSILON)
                
                with torch.no_grad():
                    # G update (block-wise EMA with curvature memory)
                    for b, act_stat in enumerate(activ_mean):
                        act_abs_mean = act_stat.mean()
                        core.G[b] = core.G[b] * (1 - MU) + ALPHA * act_abs_mean * core.C[b]
                    
                    # C update (ACP curvature memory)
                    gamma = 2.0 - (current_loss_item / L_MAX)
                    base = max(L_MAX - current_loss_item, 0.0)
                    curvature_update = DELTA * (base ** gamma) - EPSILON * core.C
                    core.C += curvature_update
                    core.C = torch.clamp(core.C, min=0.1)
            
            # CPR diagnostics
            state = cpr_controller.update(current_loss_item)
            if state == "TRIGGERED" and step % 500 == 0:
                print(f"⚠️  CPR trigger at step {step}")
                writer.add_scalar("CPR/trigger", 1, step)
            elif state == "RESET":
                print(f"🟢 CPR reset at step {step}")
                writer.add_scalar("CPR/reset", 1, step)
            
            # Logging
            if step % LOG_INTERVAL == 0:
                writer.add_scalar("Loss/train", current_loss_item, step)
                writer.add_scalar("Loss/train_smooth", ema, step)
                writer.add_scalar("Gradients/total_norm", total_norm.item(), step)
                writer.add_scalar("AMP/grad_scale", scaler.get_scale(), step)
                writer.add_scalar("ctx_len", core.ctx_len, step)
                
                # Per-layer weight and gradient histograms
                if step % 1000 == 0:
                    for name, param in core.named_parameters():
                        if param.requires_grad:
                            writer.add_histogram(f'Weights/{name}', param.data, step)
                            if param.grad is not None:
                                writer.add_histogram(f'Gradients/{name}', param.grad.data, step)
                
                # G logging
                if step % 1000 == 0:
                    writer.add_histogram("G/hist_all", core.G, step)
                for k in range(core.G.size(0)):
                    writer.add_scalar(f"G_block{k}/mean", core.G[k].mean().item(), step)
                    writer.add_scalar(f"G_block{k}/std", core.G[k].std().item(), step)
                
                writer.flush()
                
                # Print progress
                gpu_info = f"GPUs" if isinstance(model, torch.nn.DataParallel) else "GPU"
                print(f"Step {step}/{MAX_STEPS} | Using {torch.cuda.device_count()} {gpu_info} | Loss {current_loss_item:.4f} | EMA {ema:.4f} | CTX {core.ctx_len}")
            
            # Learning rate / optimizer parameter scheduler based on loss plateau
            if len(loss_history) >= PLATEAU_PATIENCE and step % PLATEAU_PATIENCE == 0:
                slope = calculate_slope(loss_history[-PLATEAU_PATIENCE:])
                writer.add_scalar("Scheduler/loss_slope", slope, step)
                
                if slope > PLATEAU_SLOPE_THRESHOLD:  # Plateau detected
                    plateau_msg = f"Plateau detected at step {step}. Slope: {slope:.5f}."
                    
                    if isinstance(optimizer, ARPOptimizer):
                        if uses_direct_attributes:
                            old_alpha = optimizer.alpha
                            old_mu = optimizer.mu
                            optimizer.alpha *= 0.7
                            optimizer.mu *= 1.05
                            
                            print(f"{plateau_msg} ARPOptimizer: α {old_alpha:.4f}→{optimizer.alpha:.4f}, μ {old_mu:.4f}→{optimizer.mu:.4f}")
                            
                            writer.add_text("Scheduler/plateau_event", f"Plateau: α→{optimizer.alpha:.4f}, μ→{optimizer.mu:.4f}", step)
                            writer.add_scalar("Scheduler/ARPO_alpha", optimizer.alpha, step)
                            writer.add_scalar("Scheduler/ARPO_mu", optimizer.mu, step)
                        elif len(optimizer.param_groups) > 0 and 'alpha' in optimizer.param_groups[0] and 'mu' in optimizer.param_groups[0]:
                            old_alpha = optimizer.param_groups[0]['alpha']
                            old_mu = optimizer.param_groups[0]['mu']
                            optimizer.param_groups[0]['alpha'] *= 0.7
                            optimizer.param_groups[0]['mu'] *= 1.05
                            
                            print(f"{plateau_msg} ARPOptimizer: α {old_alpha:.4f}→{optimizer.param_groups[0]['alpha']:.4f}, μ {old_mu:.4f}→{optimizer.param_groups[0]['mu']:.4f}")
                            
                            writer.add_text("Scheduler/plateau_event", f"Plateau: α→{optimizer.param_groups[0]['alpha']:.4f}, μ→{optimizer.param_groups[0]['mu']:.4f}", step)
                            writer.add_scalar("Scheduler/ARPO_alpha", optimizer.param_groups[0]['alpha'], step)
                            writer.add_scalar("Scheduler/ARPO_mu", optimizer.param_groups[0]['mu'], step)
                        else:
                            print(f"{plateau_msg} Couldn't update ARPOptimizer parameters (attributes not found)")
                            writer.add_text("Scheduler/plateau_event", "Plateau detected but couldn't update optimizer params", step)
                    else:
                        print(f"{plateau_msg} (Non-ARP optimizer)")
                        writer.add_text("Scheduler/plateau_event", f"Plateau detected. Slope: {slope:.5f}", step)
                    
                    loss_history.clear()
            
            # Checkpointing
            if step > 0 and step % CKPT_INTERVAL == 0:
                # Save checkpoint
                fn = CKPT_DIR / f"gpt2_dataparallel_step{step}_{int(time.time())}.pth"
                save_obj = {
                    "model_state_dict": core.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    "step": step,
                    "DELTA": DELTA,
                    "EPSILON": EPSILON
                }
                torch.save(save_obj, fn)
                
                # Delete old checkpoints
                old_ckpts = sorted(CKPT_DIR.glob("gpt2_dataparallel_step*.pth"))[:-MAX_BACKUPS]
                for f in old_ckpts:
                    f.unlink(missing_ok=True)
                
                print(f"💾 Checkpoint saved @ {step} → {fn}")
                
                # Generate sample text
                sample_text = core.generate(GEN_PROMPT, temp=0.8, top_k=50)
                writer.add_text("Sample", sample_text, step)
                print(f"Sample: {sample_text}")
            
            step += 1
            if step >= MAX_STEPS:
                break
        
        if step >= MAX_STEPS:
            break
    
    # Cleanup
    writer.flush()
    writer.close()
    print("🎉 Training loop finished")
    print("Logging to:", LOG_DIR)
    print("✅ TensorBoard logs flushed and closed")

# ── Parse arguments and initialize variables -------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Windows-compatible DataParallel training")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")
    return parser.parse_args()

# Store initial hyperparameter values as function attributes for synchronization
main_worker.DELTA = INITIAL_DELTA
main_worker.EPSILON = INITIAL_EPSILON

if __name__ == "__main__":
    args = get_args()
    
    # Check CUDA availability first
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    
    if not cuda_available:
        print("WARNING: CUDA is not available. Falling back to CPU training.")
        num_gpus = 1  # Force single CPU mode
    else:
        print(f"CUDA is available with {gpu_count} GPU(s).")
        num_gpus = min(args.num_gpus, gpu_count)
    
    # Set use_data_parallel to true to force DataParallel mode
    args.use_data_parallel = True
    
    # Launch training
    print(f"Starting training with {num_gpus} {'GPUs' if cuda_available else 'CPU worker(s)'}")
    
    # Always use a single process with DataParallel
    main_worker(0, num_gpus, args)
