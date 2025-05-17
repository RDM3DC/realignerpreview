\
# ------------------------------------------------------------
# realignr_ddp.py — DDP version with integrated Meta-Controller
# ------------------------------------------------------------
#  • To be launched with torchrun
#  • Meta-controller for ARPOptimizer runs in rank-0 thread
#  • Dynamic context schedule: 1_024 → 2_048 → 4_096 tokens
#  • AMP, TensorBoard, auto-rotating checkpoints (rank 0 only)
# ------------------------------------------------------------

import os
import time
import glob
import math
import json
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ── local utilities --------------------------------------------------
from optimizers.arp_optimizer import ARPOptimizer
from meta_controller import MetaController as LogOnlyMetaController # Renamed to avoid confusion
from meta_controller import CPRController
from action import ActionTracker

# ── CONSTANTS --------------------------------------------------------
CURVATURE_MIN_THRESHOLD = 0.15
L_MAX = 10.0
BASE_DIR   = Path(__file__).resolve().parent
STEP_START = 0
# LOG_DIR will be set in main_worker by rank 0
CKPT_DIR   = BASE_DIR / "checkpoints"
RESUME_CKPT = None

MAX_STEPS  = 300_000
CKPT_INTERVAL = 2_000
LOG_INTERVAL  = 50
MAX_BACKUPS   = 8

# MetaLoop constants
META_LOOP_CHECK_INTERVAL_STEPS = 200 # How often MetaLoop checks loss, in training steps
PLATEAU_PATIENCE_METALOOP = 800 # Window size for MetaLoop's loss slope calculation
PLATEAU_SLOPE_THRESHOLD_METALOOP = -0.001

# Initial values for dynamic hyperparameters (will be broadcast by rank 0)
# These are Python variables, DDP sync needs to be handled manually
# We'll use torch tensors for broadcasting them.
# Initial values are set here, but rank 0 will be the source of truth after any modifications.
INITIAL_DELTA = 0.01
INITIAL_EPSILON = 0.001
INITIAL_ALPHA, INITIAL_MU = 0.01, 0.001


SEQ_LEN   = 1_024
CTX_START = 1_024
BATCH_SIZE_PER_GPU = 2 # Effective batch size = BATCH_SIZE_PER_GPU * world_size
CONTEXT_SCHEDULE = [
    (50_000, 2_048),
    (120_000, 4_096),
]

GEN_PROMPT = "The meaning of life is"
GEN_LEN    = 50

# ── TOKENIZER / DATA HELPERS ----------------------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_and_chunk(ex):
    ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
    flat = list(chain.from_iterable(ids))
    return {"input_ids": [flat[i:i+SEQ_LEN] for i in range(0, len(flat), SEQ_LEN)]}

def get_packed_wikitext103(rank, world_size, split="train[:2%]", batch_size_per_gpu=BATCH_SIZE_PER_GPU):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    ds = ds.map(tokenize_and_chunk, batched=True, remove_columns=["text"])
    ds = ds.filter(lambda e: len(e["input_ids"]) == SEQ_LEN).with_format("torch")
    
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=split.startswith("train"), drop_last=True)
    return DataLoader(ds, batch_size=batch_size_per_gpu, sampler=sampler, num_workers=2, pin_memory=True)

# ── MODEL ------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
    def forward(self, x, mask):
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        self.last_out = x.detach() # Not strictly needed for DDP if activ_mean is constructed carefully
        return x

class RealignRGPT2(nn.Module):
    def __init__(self, vocab, ctx_len):
        super().__init__()
        dim = 768
        self.ctx_len  = ctx_len
        self.tok_emb  = nn.Embedding(vocab, dim)
        self.pos_emb  = nn.Parameter(torch.zeros(1, ctx_len, dim))
        self.blocks   = nn.ModuleList(Block(dim, 12) for _ in range(4)) # 4 blocks for GPT-2 small
        self.ln_f     = nn.LayerNorm(dim)
        self.head     = nn.Linear(dim, vocab, bias=False)

        g_init = torch.zeros(len(self.blocks), dim)
        self.register_buffer("G", g_init, persistent=True)
        c_init = torch.ones(len(self.blocks), dim)
        self.register_buffer("C", c_init, persistent=True)

    def expand_ctx(self, new_len, rank=0): # Add rank for conditional printing
        if new_len <= self.ctx_len: return
        old_device = self.pos_emb.device
        old_data = self.pos_emb.data
        new_pos_emb_data = torch.zeros(1, new_len, old_data.size(2), device=old_device)
        new_pos_emb_data[:, :self.ctx_len] = old_data
        nn.init.trunc_normal_(new_pos_emb_data[:, self.ctx_len:], std=0.02)
        self.pos_emb = nn.Parameter(new_pos_emb_data)
        self.ctx_len = new_len
        if rank == 0:
            print(f"🔁 context window → {new_len}")

    def forward(self, idx):
        B,T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T]
        # Create attention mask once and pass it to blocks
        # This mask should be on the same device as x
        mask = torch.triu(torch.ones(T,T,device=idx.device)*float('-inf'), 1)

        block_activ_means = [] # list of (d_model,) tensors
        for blk in self.blocks:
            x = blk(x, mask)
            block_activ_means.append(x.abs().mean(dim=(0, 1))) # Mean over Batch and Seq_Len -> (d_model,)
        
        return self.head(self.ln_f(x)), block_activ_means

    @torch.no_grad()
    def generate(self, prompt, max_len=GEN_LEN, temp=1.0, top_k=0): # Should be called on model.module by rank 0
        self.eval()
        ids = tokenizer.encode(prompt, return_tensors="pt").to(self.head.weight.device)
        for _ in range(max_len):
            logits,_ = self(ids[:, -self.ctx_len:]) # self here is model.module
            logits = logits[:,-1,:]/temp
            if top_k > 0 :
                topv, topi = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('inf'))
                logits.scatter_(1, topi, topv)
            next_tok = torch.multinomial(torch.softmax(logits,-1), 1)
            ids = torch.cat([ids,next_tok],-1)
        self.train() # return to train mode
        return tokenizer.decode(ids[0], skip_special_tokens=True)

# ── DDP Setup & Cleanup -----------------------------------------------
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Ensure this port is free
    # Initialize the process group
    # backend='nccl' for NVIDIA GPUs, 'gloo' for CPU or other GPUs
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank) # rank is used as local_rank here

def cleanup_ddp():
    dist.destroy_process_group()

# ── Helper function for loss slope calculation (used by MetaLoop) -----
def calculate_slope(loss_window):
    if not loss_window or len(loss_window) < 2:
        return 0
    y = torch.tensor(loss_window, dtype=torch.float32)
    x = torch.arange(len(y), dtype=torch.float32)
    N = len(x)
    sum_xy = torch.sum(x * y); sum_x = torch.sum(x); sum_y = torch.sum(y); sum_x_sq = torch.sum(x**2)
    denominator = N * sum_x_sq - sum_x**2
    return (N * sum_xy - sum_x * sum_y) / denominator if denominator != 0 else 0

# ── Meta-Controller Loop (runs in a thread on Rank 0) ----------------
def meta_loop_fn(rank, optimizer, writer, stop_event, meta_loop_queue, initial_alpha, initial_mu):
    if rank != 0: return

    current_alpha = initial_alpha
    current_mu = initial_mu
    # Ensure optimizer on rank 0 has these initial values
    if isinstance(optimizer, ARPOptimizer):
        optimizer.alpha = current_alpha
        optimizer.mu = current_mu

    loss_q = deque(maxlen=PLATEAU_PATIENCE_METALOOP)
    meta_logger = LogOnlyMetaController() # For meta.log() functionality

    print(f"[MetaLoop Rank 0] Started. α={current_alpha:.4f}, μ={current_mu:.4f}. Checking every ~{META_LOOP_CHECK_INTERVAL_STEPS} steps.")

    while not stop_event.is_set():
        try:
            data = meta_loop_queue.get(timeout=1.0) # Wait for data from main loop
            step = data['step']
            ema_loss = data['ema_loss']
            loss_q.append(ema_loss)

            if len(loss_q) == PLATEAU_PATIENCE_METALOOP and step % META_LOOP_CHECK_INTERVAL_STEPS == 0 :
                slope = calculate_slope(list(loss_q))
                writer.add_scalar("MetaLoop/loss_slope", slope, step)

                if slope > PLATEAU_SLOPE_THRESHOLD_METALOOP: # Plateau detected
                    if isinstance(optimizer, ARPOptimizer):
                        old_alpha, old_mu = current_alpha, current_mu
                        current_alpha *= 0.7
                        current_mu *= 1.05
                        # current_mu = min(current_mu, SOME_MAX_MU_VALUE) # Optional: cap mu

                        optimizer.alpha = current_alpha # Update rank 0's optimizer
                        optimizer.mu = current_mu

                        meta_logger.log(f"[MetaLoop Rank 0] Plateau at step {step}. Slope: {slope:.5f}. ARPO: α {old_alpha:.4f}→{current_alpha:.4f}, μ {old_mu:.4f}→{current_mu:.4f}")
                        writer.add_text("MetaLoop/plateau_event", f"Plateau: α→{current_alpha:.4f}, μ→{current_mu:.4f}", step)
                        writer.add_scalar("MetaLoop/ARPO_alpha", current_alpha, step)
                        writer.add_scalar("MetaLoop/ARPO_mu", current_mu, step)
                        
                        # Broadcast new alpha and mu to other ranks
                        alpha_mu_tensor = torch.tensor([current_alpha, current_mu], dtype=torch.float64, device=f'cuda:{rank}')
                        dist.broadcast(alpha_mu_tensor, src=0)
                        
                        loss_q.clear() # Reset history
                    else:
                        meta_logger.log(f"[MetaLoop Rank 0] Plateau detected at step {step}. Slope: {slope:.5f}. (Non-ARP optimizer)")
                        writer.add_text("MetaLoop/plateau_event", f"Plateau detected. Slope: {slope:.5f}", step)
                writer.flush() # Ensure MetaLoop specific logs are written
        except queue.Empty:
            if stop_event.is_set(): break
            continue # No new data, continue waiting or check stop_event
        except Exception as e:
            print(f"[MetaLoop Rank 0] Error: {e}")
            if stop_event.is_set(): break
            time.sleep(5) # Avoid busy-looping on error

    print("[MetaLoop Rank 0] Stopped.")

# Define helper function for curvature variance checking (used by rank 0 for DELTA/EPSILON logic)
def curvature_variance_high(C_tensor):
    c_std = C_tensor.std().item()
    c_mean = C_tensor.mean().item()
    return c_mean > 0 and (c_std / c_mean) > 0.5


# ── MAIN WORKER FUNCTION (called by torchrun for each process) -------
def main_worker(rank, world_size):
    setup_ddp(rank, world_size)
    
    # Hyperparameters that are dynamically adjusted by rank 0 and broadcast
    # Initialize with placeholder values, will be overwritten by broadcast from rank 0
    # Using torch tensors on the correct device for dist.broadcast operations
    # Ensure dtype is float64 for precision if values are small.
    current_device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    
    # DELTA, EPSILON, ALPHA, MU
    # Rank 0 will initialize these and broadcast. Other ranks receive.
    dynamic_hyperparams = torch.tensor([INITIAL_DELTA, INITIAL_EPSILON, INITIAL_ALPHA, INITIAL_MU], dtype=torch.float64, device=current_device)
    if rank == 0:
        # Rank 0 is the source of truth for initial values if they were to be loaded or complexly initialized
        # Here, they are just the constants.
        pass # dynamic_hyperparams already has INITIAL values for rank 0
    dist.broadcast(dynamic_hyperparams, src=0)
    
    # Assign to Python variables on all ranks
    DELTA = dynamic_hyperparams[0].item()
    EPSILON = dynamic_hyperparams[1].item()
    # ALPHA and MU for optimizer are handled slightly differently:
    # optimizer is initialized with them, and MetaLoop (rank 0) updates its optimizer
    # and broadcasts. Other ranks update their optimizer upon receiving broadcast.
    # So, direct assignment of ALPHA, MU here is for non-optimizer logic if any.
    # The optimizer itself will get its alpha/mu set correctly.
    ALPHA = dynamic_hyperparams[2].item() # For G update logic
    MU = dynamic_hyperparams[3].item()    # For G update logic


    # Create LOG_DIR and CKPT_DIR (rank 0 primarily, others ensure it exists)
    # LOG_DIR needs a timestamp, rank 0 defines it.
    log_dir_str_tensor = None
    if rank == 0:
        _log_dir_path = BASE_DIR / "runs" / f"ddp_cpr_sanity_{int(time.time())}"
        os.makedirs(_log_dir_path, exist_ok=True)
        log_dir_str = str(_log_dir_path)
        # Convert string to tensor for broadcast
        log_dir_bytes = log_dir_str.encode('utf-8')
        log_dir_tensor = torch.tensor(list(log_dir_bytes), dtype=torch.uint8, device=current_device)
        log_dir_len_tensor = torch.tensor([len(log_dir_bytes)], dtype=torch.int64, device=current_device)
        dist.broadcast(log_dir_len_tensor, src=0)
        dist.broadcast(log_dir_tensor, src=0) # Broadcast the tensor itself
        LOG_DIR = Path(log_dir_str)
    else:
        log_dir_len_tensor = torch.empty(1, dtype=torch.int64, device=current_device)
        dist.broadcast(log_dir_len_tensor, src=0)
        log_dir_tensor_shape = (log_dir_len_tensor.item(),)
        log_dir_tensor = torch.empty(log_dir_tensor_shape, dtype=torch.uint8, device=current_device)
        dist.broadcast(log_dir_tensor, src=0)
        log_dir_str = bytes(log_dir_tensor.tolist()).decode('utf-8')
        LOG_DIR = Path(log_dir_str)

    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
    dist.barrier() # Ensure dirs are created before proceeding

    # Model setup
    base_model = RealignRGPT2(tokenizer.vocab_size, ctx_len=CTX_START).to(current_device)
    # DDP wrapper
    # find_unused_parameters can be True if some outputs of model.forward are not used in loss
    # For RealignRGPT2, activ_mean is returned but not directly in loss.
    # However, G and C updates depend on it, and G/C are buffers.
    # DDP handles buffer sync. If G/C updates are correct, this should be fine.
    # Let's start with find_unused_parameters=False, if errors, set to True.
    model = DDP(base_model, device_ids=[rank] if torch.cuda.is_available() else None, find_unused_parameters=False)
    core = model.module # To access original model methods like generate, expand_ctx, and buffers G, C

    if rank == 0:
        print(f"DDP Initialized. Rank {rank}/{world_size} on device {current_device}")
        print(f"Model initialized with context length: {core.ctx_len}")
        print(f"G shape: {core.G.shape}, C shape: {core.C.shape}")

    # Optimizer (ARPOptimizer with initial ALPHA, MU)
    # Each rank initializes its own optimizer. MetaLoop on rank 0 will adjust its own
    # and then broadcast the new alpha/mu values for other ranks to apply.
    optimizer = ARPOptimizer(model.parameters(), alpha=INITIAL_ALPHA, mu=INITIAL_MU)
    if rank == 0:
         print(f"🔀 Starting with ARP (initial α={INITIAL_ALPHA}, μ={INITIAL_MU}) at step {STEP_START}")
    
    scaler = GradScaler() # AMP

    # Checkpoint loading (simplified for fresh start)
    step = STEP_START
    if rank == 0:
        print("🚀 Starting fresh training run from scratch.")
        # print("First token weights L2:", core.tok_emb.weight[0].norm().item()) # If needed

    # Controllers and Loggers (Rank 0 only for SummaryWriter and primary logging)
    writer = None
    meta_loop_thread = None
    meta_loop_stop_event = None
    meta_loop_q = None
    cpr_controller = None
    log_only_meta_controller = None # For meta.log()
    action_tracker_obj = None

    if rank == 0:
        writer = SummaryWriter(str(LOG_DIR), purge_step=STEP_START)
        print(f"▶️  Training starts at step {step}, ctx={core.ctx_len}")
        print(f"[TensorBoard] Log directory: {LOG_DIR}")
        print(f"[TensorBoard] Run: tensorboard --logdir \"{LOG_DIR}\"")
        
        cpr_controller = CPRController(epsilon=1e-3, reset_patience=500)
        log_only_meta_controller = LogOnlyMetaController()
        action_tracker_obj = ActionTracker(λ1=0.15) # If used

        # MetaLoop setup for Rank 0
        meta_loop_q = queue.Queue()
        meta_loop_stop_event = threading.Event()
        # Pass initial_alpha and initial_mu that optimizer on rank 0 is using
        meta_loop_thread = threading.Thread(target=meta_loop_fn, 
                                            args=(rank, optimizer, writer, meta_loop_stop_event, meta_loop_q, 
                                                  optimizer.alpha, optimizer.mu))
        meta_loop_thread.start()

    # Data Loader with DistributedSampler
    train_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:2%]") # Load once
    train_dataset_processed = train_dataset.map(tokenize_and_chunk, batched=True, remove_columns=["text"])
    train_dataset_filtered = train_dataset_processed.filter(lambda e: len(e["input_ids"]) == SEQ_LEN).with_format("torch")
    
    train_sampler = DistributedSampler(train_dataset_filtered, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset_filtered, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler, num_workers=2, pin_memory=True)
    
    # Training loop variables
    ema = None
    ctx_idx = 0
    last_curv_warn_step = -100 # For rank 0 warnings

    # Main training loop
    for epoch in range(MAX_STEPS // len(train_loader) + 1): # Iterate for enough epochs
        train_sampler.set_epoch(epoch) # Important for shuffling with DDP
        if rank == 0 and epoch > 0 : print(f"Starting Epoch {epoch}")

        for i, batch in enumerate(train_loader):
            if step >= MAX_STEPS: break

            # Dynamic DELTA / EPSILON adjustment (Rank 0 decides, broadcasts)
            # This needs to be done *before* they are used in ACP update.
            # Ensure core.C is up-to-date on rank 0 (DDP syncs buffers)
            if rank == 0:
                c_mean_val = core.C.mean().item() # core.C is synchronized by DDP
                new_delta, new_epsilon = DELTA, EPSILON # Start with current values

                if c_mean_val < CURVATURE_MIN_THRESHOLD:
                    new_delta *= 1.05
                    log_only_meta_controller.log(f"Increasing curvature learning rate δ to {new_delta:.4f} at step {step}")
                
                if curvature_variance_high(core.C): # core.C is synchronized
                    new_epsilon *= 0.95
                    log_only_meta_controller.log(f"Reducing curvature decay ε to {new_epsilon:.5f} at step {step}")

                # Update dynamic_hyperparams tensor for broadcast
                dynamic_hyperparams[0] = new_delta
                dynamic_hyperparams[1] = new_epsilon
                # Note: alpha/mu are handled by MetaLoop's broadcast
            
            # Broadcast DELTA and EPSILON from rank 0 to all other ranks
            dist.broadcast(dynamic_hyperparams, src=0) # This broadcasts all 4, but only DELTA/EPSILON are set here by rank 0 main loop
            DELTA = dynamic_hyperparams[0].item()
            EPSILON = dynamic_hyperparams[1].item()
            # ALPHA and MU from this broadcast are the ones from the *start* of this step,
            # MetaLoop might update them based on *previous* step's loss.
            # The optimizer alpha/mu are updated separately when MetaLoop broadcasts.

            # Context expansion (all ranks must do this to keep model consistent)
            if ctx_idx < len(CONTEXT_SCHEDULE) and step >= CONTEXT_SCHEDULE[ctx_idx][0]:
                target_ctx = CONTEXT_SCHEDULE[ctx_idx][1]
                # core is model.module, so all ranks call expand_ctx on their local module instance
                if target_ctx > core.ctx_len: # Check on each rank
                    core.expand_ctx(target_ctx, rank=rank) # Pass rank for conditional print
                elif rank == 0: # Log skip only on rank 0
                    print(f"Skipping context schedule entry {CONTEXT_SCHEDULE[ctx_idx]} as target length {target_ctx} is not greater than current {core.ctx_len}")
                ctx_idx += 1
            dist.barrier() # Ensure all ranks have expanded context if needed

            input_ids = batch["input_ids"].to(current_device)
            labels = input_ids.clone()

            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                logits, local_block_activ_means = model(input_ids) # model is DDP wrapped
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
                    labels[:, 1:].reshape(-1)
                )
            
            # Aggregate loss for reporting (optional, but good for consistent logging)
            # loss_val_for_reporting = loss.clone().detach()
            # dist.all_reduce(loss_val_for_reporting, op=dist.ReduceOp.AVG)
            # current_loss_item = loss_val_for_reporting.item()
            current_loss_item = loss.item() # Use per-rank loss for now for EMA, MetaLoop gets rank 0's EMA

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip per-rank grads
            scaler.step(optimizer)
            scaler.update()

            # EMA-based G update and ACP curvature memory update
            # Needs globally averaged activation means
            if isinstance(optimizer, ARPOptimizer): # Check if ARP is active
                # 1. AllReduce local_block_activ_means to get global averages
                global_block_activ_means = []
                for local_am_tensor in local_block_activ_means:
                    # Each local_am_tensor is (d_model,)
                    reduced_am_tensor = local_am_tensor.clone().detach() # Ensure it's a leaf and separate
                    dist.all_reduce(reduced_am_tensor, op=dist.ReduceOp.AVG)
                    global_block_activ_means.append(reduced_am_tensor)
                
                # 2. All ranks perform G and C updates using these global means
                # This ensures core.G and core.C (model buffers) remain synchronized.
                with torch.no_grad():
                    # G update
                    for b_idx, globally_avg_act_stat_tensor in enumerate(global_block_activ_means):
                        # globally_avg_act_stat_tensor is (d_model,)
                        # The original code took act_stat.mean() to get a scalar.
                        # act_abs_mean = globally_avg_act_stat_tensor.mean() # This is a scalar
                        # Let's re-check: original code `act_abs_mean = act_stat.mean()` where act_stat was `x.abs().mean((0,1))`
                        # So `local_block_activ_means` already contains per-block (d_model,) tensors that are means over B,T.
                        # The G update used `ALPHA * act_abs_mean * core.C[b]`. If act_abs_mean is scalar, C[b] is (d_model), this is fine.
                        # If `act_abs_mean` was intended to be the (d_model) tensor itself, then it's element-wise.
                        # The prompt's example `activ_mean.append(x.abs().mean((0, 1))) # (d_model,)`
                        # And `core.G[b] = core.G[b] * (1 - MU) + ALPHA * act_abs_mean * core.C[b]`
                        # If act_abs_mean is (d_model), then this is element-wise. This seems more likely.
                        # So, globally_avg_act_stat_tensor is the (d_model,) tensor to use.
                        core.G[b_idx] = core.G[b_idx] * (1 - MU) + ALPHA * globally_avg_act_stat_tensor * core.C[b_idx]
                    
                    # C update (ACP)
                    # Loss for ACP should ideally be the global average loss
                    # For simplicity, using rank 0's loss for gamma calculation if not all-reduced before.
                    # Let's use current_loss_item (which is rank's loss, but rank 0 sends its to MetaLoop)
                    # For consistency in C update across ranks, it's better if all ranks use the same loss value.
                    # Let's all-reduce the loss for this specific calculation.
                    loss_for_c_update = loss.clone().detach()
                    dist.all_reduce(loss_for_c_update, op=dist.ReduceOp.AVG)
                    loss_item_for_c = loss_for_c_update.item()

                    gamma = 2.0 - (loss_item_for_c / L_MAX) # Use L_MAX from constants
                    base = max(L_MAX - loss_item_for_c, 0.0)
                    curvature_update = DELTA * (base ** gamma) - EPSILON * core.C
                    core.C += curvature_update
                    core.C = torch.clamp(core.C, min=0.1)

            # Rank 0: Logging, EMA, CPR, send to MetaLoop
            if rank == 0:
                ema = current_loss_item if ema is None else 0.99*ema + 0.01*current_loss_item
                
                # Send data to MetaLoop
                if step % (META_LOOP_CHECK_INTERVAL_STEPS // 4) == 0: # Send more frequently than MetaLoop checks
                     try:
                        meta_loop_q.put_nowait({'step': step, 'ema_loss': ema})
                     except queue.Full:
                        pass # Skip if queue is full, MetaLoop will catch up

                if step % LOG_INTERVAL == 0:
                    writer.add_scalar("Loss/train_rank0", current_loss_item, step)
                    writer.add_scalar("Loss/train_smooth_rank0", ema, step)
                    writer.add_scalar("Gradients/total_norm_rank0", total_norm.item(), step)
                    writer.add_scalar("AMP/grad_scale", scaler.get_scale(), step)
                    writer.add_scalar("Hyperparams/DELTA", DELTA, step)
                    writer.add_scalar("Hyperparams/EPSILON", EPSILON, step)
                    writer.add_scalar("Hyperparams/ARPO_alpha_main", optimizer.alpha if isinstance(optimizer, ARPOptimizer) else -1, step)
                    writer.add_scalar("Hyperparams/ARPO_mu_main", optimizer.mu if isinstance(optimizer, ARPOptimizer) else -1, step)
                    writer.add_scalar("ctx_len", core.ctx_len, step)

                    # Curvature C logging
                    writer.add_scalar('Curvature/C_mean', core.C.mean().item(), step)
                    writer.add_scalar('Curvature/C_std', core.C.std().item(), step)
                    if step % (LOG_INTERVAL * 10) == 0: # Less frequent histogram
                         writer.add_histogram('Curvature/C_hist', core.C, step)
                    # for k_c in range(core.C.size(0)):
                    #    writer.add_scalar(f'Curvature/C_block{k_c}_mean', core.C[k_c].mean().item(), step)
                    #    writer.add_scalar(f'Curvature/C_block{k_c}_std', core.C[k_c].std().item(), step)
                    
                    # G logging
                    if step % (LOG_INTERVAL * 10) == 0: # Less frequent histogram
                        writer.add_histogram("G/hist_all", core.G, step)
                    # for k_g in range(core.G.size(0)):
                    #    writer.add_scalar(f"G_block{k_g}/mean", core.G[k_g].mean().item(), step)
                    #    writer.add_scalar(f"G_block{k_g}/std", core.G[k_g].std().item(), step)


                    # Per-layer weight and gradient histograms (less frequently)
                    if step % (LOG_INTERVAL * 20) == 0:
                        for name, param in core.named_parameters(): # core is model.module
                            if param.requires_grad:
                                writer.add_histogram(f'Weights/{name}', param.data, step)
                                if param.grad is not None: # Grads are per-rank, this is rank 0's grad
                                    writer.add_histogram(f'Gradients_rank0/{name}', param.grad.data, step)
                    
                    writer.flush()
                    print(f"Epoch {epoch} | Step {step}/{MAX_STEPS} | Rank {rank} | Loss {current_loss_item:.4f} | EMA {ema:.4f} | CTX {core.ctx_len} | GradNorm {total_norm:.4f}")

                # CPR diagnostics (rank 0 only)
                if cpr_controller:
                    cpr_state = cpr_controller.update(current_loss_item) # Using rank 0 loss
                    if cpr_state == "TRIGGERED" and step % 500 == 0: # Log less frequently
                        print(f"⚠️  CPR trigger at step {step} on Rank 0")
                        writer.add_scalar("CPR/trigger", 1, step)
                    elif cpr_state == "RESET":
                        print(f"🟢 CPR reset at step {step} on Rank 0")
                        writer.add_scalar("CPR/reset", 1, step)
                
                # Curvature mean high warning (rank 0 only)
                C_mean_threshold = 2.0
                if core.C.mean().item() > C_mean_threshold:
                    if step - last_curv_warn_step >= 100 * LOG_INTERVAL: # Much less frequent
                        print(f"⚠️ Rank 0: Curvature mean high ({core.C.mean().item():.2f}) at step {step}")
                        last_curv_warn_step = step

                # Checkpointing and generation (rank 0 only)
                if step > 0 and step % CKPT_INTERVAL == 0:
                    # Save checkpoint (rank 0 only, saving model.module state)
                    fn = CKPT_DIR / f"gpt2_ddp_step{step}_{int(time.time())}.pth"
                    save_obj = {
                        "model_state_dict": core.state_dict(), # Save module's state_dict
                        "optimizer_state_dict": optimizer.state_dict(), # Rank 0's optimizer
                        "scaler_state_dict": scaler.state_dict(), # Rank 0's scaler
                        "step": step,
                        "ctx_len": core.ctx_len,
                        "ALPHA": optimizer.alpha if isinstance(optimizer, ARPOptimizer) else ALPHA, # Current ALPHA
                        "MU": optimizer.mu if isinstance(optimizer, ARPOptimizer) else MU,       # Current MU
                        "DELTA": DELTA,   # Current DELTA
                        "EPSILON": EPSILON # Current EPSILON
                    }
                    torch.save(save_obj, fn)
                    old_ckpts = sorted(CKPT_DIR.glob("gpt2_ddp_step*.pth"))[:-MAX_BACKUPS]
                    for f_old in old_ckpts: f_old.unlink(missing_ok=True)
                    print(f"💾 Checkpoint saved @ {step} → {fn}")
                    
                    # Generate sample text
                    sample_text = core.generate(GEN_PROMPT, temp=0.8, top_k=50)
                    writer.add_text("Sample Generation", sample_text, step)
                    print(f"Sample @ {step}: {sample_text}")

            # Update optimizer parameters (alpha, mu) if MetaLoop broadcasted them
            # All ranks (including rank 0) listen for this broadcast.
            # MetaLoop on rank 0 already updated its own optimizer instance.
            # This ensures other ranks also update their optimizer instances.
            # The dynamic_hyperparams tensor is used for DELTA/EPSILON.
            # For alpha/mu, MetaLoop sends a specific 2-element tensor.
            # Let's use a separate tensor for alpha/mu broadcast from MetaLoop.
            # This needs to be non-blocking for ranks != 0 if rank 0 hasn't broadcasted yet.
            # Or, more simply, MetaLoop updates rank 0's optimizer.
            # Then, after optimizer.step(), we can broadcast rank 0's alpha/mu to all.
            # This is simpler than having MetaLoop do an extra broadcast.
            # However, the user prompt implied MetaLoop broadcasts. Let's stick to that.
            # MetaLoop (rank 0) broadcasts `alpha_mu_tensor`. All ranks need to receive it.
            if isinstance(optimizer, ARPOptimizer):
                # Prepare to receive broadcast from MetaLoop (rank 0)
                # This tensor is defined inside meta_loop_fn for rank 0.
                # Other ranks need a tensor to receive into.
                # This broadcast happens *inside* meta_loop_fn.
                # The main loop needs to check if new values were broadcast.
                # This is tricky. A simpler model:
                # Rank 0's MetaLoop modifies rank 0's optimizer.alpha/mu.
                # Then, in the main loop, rank 0 broadcasts its current optimizer.alpha/mu.
                # All other ranks receive and update.
                
                # Simpler: MetaLoop (rank 0) broadcasts. All ranks (including 0) call broadcast to receive.
                # This is how dynamic_hyperparams for DELTA/EPSILON works.
                # Let's assume MetaLoop's broadcast of alpha_mu_tensor is correctly received by other ranks.
                # This requires careful synchronization or non-blocking receives.
                
                # For now, let's assume the MetaLoop's broadcast mechanism for alpha/mu is:
                # 1. MetaLoop (rank 0) decides new alpha/mu.
                # 2. MetaLoop (rank 0) updates its *own* optimizer instance.
                # 3. MetaLoop (rank 0) broadcasts these two values.
                # 4. All ranks (including rank 0, though redundant for it) receive these values
                #    and update their *own* optimizer instances.
                # This is what `dist.broadcast(alpha_mu_tensor, src=0)` in MetaLoop does.
                # So, other ranks need to call `dist.broadcast` to receive.
                
                # This broadcast should be conditional (e.g., only if MetaLoop signaled an update).
                # This gets complex.
                # A more robust way for ARPOptimizer params:
                # After MetaLoop on rank 0 potentially changes optimizer.alpha/mu,
                # Rank 0 prepares a tensor with its current optimizer.alpha and optimizer.mu.
                # Rank 0 broadcasts this tensor. All other ranks receive and update their optimizer.
                # This happens every step, ensuring eventual consistency if MetaLoop updates.
                
                arp_params_to_sync = torch.tensor([optimizer.alpha, optimizer.mu] if isinstance(optimizer, ARPOptimizer) else [-1.0, -1.0], dtype=torch.float64, device=current_device)
                if rank == 0: # Rank 0 is the source of truth from its (potentially MetaLoop-updated) optimizer
                    if isinstance(optimizer, ARPOptimizer):
                        arp_params_to_sync[0] = optimizer.alpha
                        arp_params_to_sync[1] = optimizer.mu
                
                dist.broadcast(arp_params_to_sync, src=0) # All ranks call this
                
                if rank != 0 and isinstance(optimizer, ARPOptimizer): # Other ranks update their optimizer
                    optimizer.alpha = arp_params_to_sync[0].item()
                    optimizer.mu = arp_params_to_sync[1].item()


            step += 1
            if step >= MAX_STEPS: break
        
        if step >= MAX_STEPS: break # Break outer epoch loop too

    # End of training
    if rank == 0:
        print("🎉 Training loop finished.")
        print("Logging to:", LOG_DIR)
        if writer:
            writer.flush()
            writer.close()
            print("✅ TensorBoard logs flushed and closed on Rank 0.")
        
        if meta_loop_thread:
            print("[Main Rank 0] Signaling MetaLoop to stop...")
            meta_loop_stop_event.set()
            meta_loop_thread.join(timeout=5)
            if meta_loop_thread.is_alive():
                print("[Main Rank 0] MetaLoop thread did not join in time.")
            else:
                print("[Main Rank 0] MetaLoop thread joined.")

    cleanup_ddp()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1 # Or from env for torchrun
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE
    rank = int(os.environ.get("RANK", "0"))
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    
    # If not using torchrun, world_size might be determined differently.
    # For torchrun, it sets WORLD_SIZE.
    # The script assumes it's launched via torchrun, which sets these.
    main_worker(rank, world_size_env)

