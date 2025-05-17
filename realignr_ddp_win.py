\
# ------------------------------------------------------------
# realignr_ddp_win.py — Windows-friendly DDP version with integrated Meta-Controller
# ------------------------------------------------------------
#  • Windows-friendly DDP using torch.multiprocessing.spawn directly
#  • Meta-controller for ARPOptimizer runs in rank-0 thread
#  • Dynamic context schedule: 1_024 → 2_048 → 4_096 tokens
#  • AMP, TensorBoard, auto-rotating checkpoints
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
from datetime import timedelta

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
import torch.multiprocessing as mp

# ── local utilities --------------------------------------------------
from optimizers.arp_optimizer import ARPOptimizer
from meta_controller import MetaController as LogOnlyMetaController  # Renamed to avoid confusion
from meta_controller import CPRController
from action import ActionTracker

# ── CONSTANTS --------------------------------------------------------
CURVATURE_MIN_THRESHOLD = 0.15
L_MAX = 10.0
BASE_DIR = Path(__file__).resolve().parent
STEP_START = 0
# LOG_DIR will be set in main_worker by rank 0
CKPT_DIR = BASE_DIR / "checkpoints"
RESUME_CKPT = None

MAX_STEPS = 300_000
CKPT_INTERVAL = 2_000
LOG_INTERVAL = 50
MAX_BACKUPS = 8

# MetaLoop constants
META_LOOP_CHECK_INTERVAL_STEPS = 200  # How often MetaLoop checks loss, in training steps
PLATEAU_PATIENCE_METALOOP = 800  # Window size for MetaLoop's loss slope calculation
PLATEAU_SLOPE_THRESHOLD_METALOOP = -0.001

# Initial values for dynamic hyperparameters (will be broadcast by rank 0)
INITIAL_DELTA = 0.01
INITIAL_EPSILON = 0.001
INITIAL_ALPHA, INITIAL_MU = 0.01, 0.001

SEQ_LEN = 1_024
CTX_START = 1_024
BATCH_SIZE_PER_GPU = 2  # Effective batch size = BATCH_SIZE_PER_GPU * world_size
CONTEXT_SCHEDULE = [
    (50_000, 2_048),
    (120_000, 4_096),
]

GEN_PROMPT = "The meaning of life is"
GEN_LEN = 50

# ── TOKENIZER ------------------------------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_and_chunk(ex):
    ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
    flat = list(chain.from_iterable(ids))
    return {"input_ids": [flat[i:i+SEQ_LEN] for i in range(0, len(flat), SEQ_LEN)]}

# ── DDP Setup & Cleanup -----------------------------------------------
def setup_ddp(rank, world_size, backend='gloo'):  # Use 'gloo' for Windows!
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Ensure this port is free
    
    # Disable libuv to avoid Windows-specific errors
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['TP_SOCKET_IFNAME'] = 'lo'
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    
    # For Windows, explicitly set store parameters to avoid libuv issues
    store = dist.TCPStore(
        host_name='localhost',
        port=12355,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=timedelta(seconds=30),
        wait_for_workers=True,
    )
    
    # Initialize the process group with the custom store
    dist.init_process_group(
        backend=backend,
        store=store,
        rank=rank,
        world_size=world_size,
    )
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# ── MODEL ------------------------------------------------------------
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
        self.register_buffer("G", g_init, persistent=True)
        c_init = torch.ones(len(self.blocks), dim)
        self.register_buffer("C", c_init, persistent=True)

    def expand_ctx(self, new_len, rank=0):
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
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T]
        mask = torch.triu(torch.ones(T, T, device=idx.device)*float('-inf'), 1)

        block_activ_means = []
        for blk in self.blocks:
            x = blk(x, mask)
            block_activ_means.append(x.abs().mean(dim=(0, 1)))

        return self.head(self.ln_f(x)), block_activ_means

    @torch.no_grad()
    def generate(self, prompt, max_len=GEN_LEN, temp=1.0, top_k=0):
        self.eval()
        ids = tokenizer.encode(prompt, return_tensors="pt").to(self.head.weight.device)
        for _ in range(max_len):
            logits, _ = self(ids[:, -self.ctx_len:])
            logits = logits[:, -1, :]/temp
            if top_k > 0:
                topv, topi = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('inf'))
                logits.scatter_(1, topi, topv)
            next_tok = torch.multinomial(torch.softmax(logits, -1), 1)
            ids = torch.cat([ids, next_tok], -1)
        self.train()
        return tokenizer.decode(ids[0], skip_special_tokens=True)

# ── Helper function for loss slope calculation ------------------------
def calculate_slope(loss_window):
    if not loss_window or len(loss_window) < 2:
        return 0
    y = torch.tensor(loss_window, dtype=torch.float32)
    x = torch.arange(len(y), dtype=torch.float32)
    N = len(x)
    sum_xy = torch.sum(x * y)
    sum_x = torch.sum(x)
    sum_y = torch.sum(y)
    sum_x_sq = torch.sum(x**2)
    denominator = N * sum_x_sq - sum_x**2
    return (N * sum_xy - sum_x * sum_y) / denominator if denominator != 0 else 0

# ── Meta-Controller Loop (runs in a thread on Rank 0) -----------------
def meta_loop_fn(rank, optimizer, writer, stop_event, meta_loop_queue, initial_alpha, initial_mu):
    if rank != 0: return

    current_alpha = initial_alpha
    current_mu = initial_mu
    if isinstance(optimizer, ARPOptimizer):
        optimizer.alpha = current_alpha
        optimizer.mu = current_mu

    loss_q = deque(maxlen=PLATEAU_PATIENCE_METALOOP)
    meta_logger = LogOnlyMetaController()

    print(f"[MetaLoop Rank 0] Started. α={current_alpha:.4f}, μ={current_mu:.4f}. Checking every ~{META_LOOP_CHECK_INTERVAL_STEPS} steps.")

    while not stop_event.is_set():
        try:
            data = meta_loop_queue.get(timeout=1.0)
            step = data['step']
            ema_loss = data['ema_loss']
            loss_q.append(ema_loss)

            if len(loss_q) == PLATEAU_PATIENCE_METALOOP and step % META_LOOP_CHECK_INTERVAL_STEPS == 0:
                slope = calculate_slope(list(loss_q))
                writer.add_scalar("MetaLoop/loss_slope", slope, step)

                if slope > PLATEAU_SLOPE_THRESHOLD_METALOOP:  # Plateau detected
                    if isinstance(optimizer, ARPOptimizer):
                        old_alpha, old_mu = current_alpha, current_mu
                        current_alpha *= 0.7
                        current_mu *= 1.05

                        optimizer.alpha = current_alpha
                        optimizer.mu = current_mu

                        meta_logger.log(f"[MetaLoop Rank 0] Plateau at step {step}. Slope: {slope:.5f}. ARPO: α {old_alpha:.4f}→{current_alpha:.4f}, μ {old_mu:.4f}→{current_mu:.4f}")
                        writer.add_text("MetaLoop/plateau_event", f"Plateau: α→{current_alpha:.4f}, μ→{current_mu:.4f}", step)
                        writer.add_scalar("MetaLoop/ARPO_alpha", current_alpha, step)
                        writer.add_scalar("MetaLoop/ARPO_mu", current_mu, step)

                        # Broadcast new alpha and mu to other ranks
                        alpha_mu_tensor = torch.tensor([current_alpha, current_mu], dtype=torch.float64, device=f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
                        dist.broadcast(alpha_mu_tensor, src=0)

                        loss_q.clear()
                    else:
                        meta_logger.log(f"[MetaLoop Rank 0] Plateau detected at step {step}. Slope: {slope:.5f}. (Non-ARP optimizer)")
                        writer.add_text("MetaLoop/plateau_event", f"Plateau detected. Slope: {slope:.5f}", step)
                writer.flush()
        except queue.Empty:
            if stop_event.is_set(): break
            continue
        except Exception as e:
            print(f"[MetaLoop Rank 0] Error: {e}")
            if stop_event.is_set(): break
            time.sleep(5)

    print("[MetaLoop Rank 0] Stopped.")

# Define helper function for curvature variance checking
def curvature_variance_high(C_tensor):
    c_std = C_tensor.std().item()
    c_mean = C_tensor.mean().item()
    return c_mean > 0 and (c_std / c_mean) > 0.5

# ── MAIN WORKER FUNCTION ---------------------------------------------
def main_worker(rank, world_size, args):
    # Setup the environment for this process
    backend = args.backend
    setup_ddp(rank, world_size, backend)
    
    # Hyperparameters that are dynamically adjusted by rank 0 and broadcast
    current_device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    
    # Initial dynamic hyperparameters
    dynamic_hyperparams = torch.tensor([INITIAL_DELTA, INITIAL_EPSILON, INITIAL_ALPHA, INITIAL_MU], 
                                      dtype=torch.float64, device=current_device)
    dist.broadcast(dynamic_hyperparams, src=0)
    
    # Assign to Python variables on all ranks
    DELTA = dynamic_hyperparams[0].item()
    EPSILON = dynamic_hyperparams[1].item()
    ALPHA = dynamic_hyperparams[2].item()
    MU = dynamic_hyperparams[3].item()

    # Create LOG_DIR and CKPT_DIR
    log_dir_str_tensor = None
    if rank == 0:
        _log_dir_path = BASE_DIR / "runs" / f"ddp_win_cpr_sanity_{int(time.time())}"
        os.makedirs(_log_dir_path, exist_ok=True)
        log_dir_str = str(_log_dir_path)
        log_dir_bytes = log_dir_str.encode('utf-8')
        log_dir_tensor = torch.tensor(list(log_dir_bytes), dtype=torch.uint8, device=current_device)
        log_dir_len_tensor = torch.tensor([len(log_dir_bytes)], dtype=torch.int64, device=current_device)
        
        dist.broadcast(log_dir_len_tensor, src=0)
        log_dir_tensor_full = torch.zeros(log_dir_len_tensor.item(), dtype=torch.uint8, device=current_device)
        log_dir_tensor_full[:len(log_dir_bytes)] = log_dir_tensor
        dist.broadcast(log_dir_tensor_full, src=0)
        
        LOG_DIR = Path(log_dir_str)
    else:
        log_dir_len_tensor = torch.zeros(1, dtype=torch.int64, device=current_device)
        dist.broadcast(log_dir_len_tensor, src=0)
        log_dir_tensor_full = torch.zeros(log_dir_len_tensor.item(), dtype=torch.uint8, device=current_device)
        dist.broadcast(log_dir_tensor_full, src=0)
        log_dir_str = bytes(log_dir_tensor_full.tolist()).decode('utf-8')
        LOG_DIR = Path(log_dir_str)

    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
    dist.barrier()

    # Model setup
    base_model = RealignRGPT2(tokenizer.vocab_size, ctx_len=CTX_START).to(current_device)
    # In Windows with gloo, find_unused_parameters=True is sometimes needed
    model = DDP(base_model, device_ids=[rank] if torch.cuda.is_available() else None, 
                find_unused_parameters=(args.backend == 'gloo'))
    core = model.module

    if rank == 0:
        print(f"DDP Initialized with {args.backend} backend. Rank {rank}/{world_size} on device {current_device}")
        print(f"Model initialized with context length: {core.ctx_len}")
        print(f"G shape: {core.G.shape}, C shape: {core.C.shape}")

    # Optimizer
    optimizer = ARPOptimizer(model.parameters(), alpha=INITIAL_ALPHA, mu=INITIAL_MU)
    if rank == 0:
        print(f"🔀 Starting with ARP (initial α={INITIAL_ALPHA}, μ={INITIAL_MU}) at step {STEP_START}")
    
    scaler = GradScaler()  # AMP

    # Step counter
    step = STEP_START
    if rank == 0:
        print("🚀 Starting fresh training run from scratch.")

    # Controllers and Loggers
    writer = None
    meta_loop_thread = None
    meta_loop_stop_event = None
    meta_loop_q = None
    cpr_controller = None
    log_only_meta_controller = None
    action_tracker_obj = None

    if rank == 0:
        writer = SummaryWriter(str(LOG_DIR), purge_step=STEP_START)
        print(f"▶️  Training starts at step {step}, ctx={core.ctx_len}")
        print(f"[TensorBoard] Log directory: {LOG_DIR}")
        print(f"[TensorBoard] Run: tensorboard --logdir \"{LOG_DIR}\"")
        
        cpr_controller = CPRController(epsilon=1e-3, reset_patience=500)
        log_only_meta_controller = LogOnlyMetaController()
        action_tracker_obj = ActionTracker(λ1=0.15)

        # MetaLoop setup for Rank 0
        meta_loop_q = queue.Queue()
        meta_loop_stop_event = threading.Event()
        meta_loop_thread = threading.Thread(
            target=meta_loop_fn, 
            args=(rank, optimizer, writer, meta_loop_stop_event, meta_loop_q, optimizer.alpha, optimizer.mu)
        )
        meta_loop_thread.start()

    # Data Loader with DistributedSampler
    train_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:2%]")
    train_dataset_processed = train_dataset.map(tokenize_and_chunk, batched=True, remove_columns=["text"])
    train_dataset_filtered = train_dataset_processed.filter(lambda e: len(e["input_ids"]) == SEQ_LEN).with_format("torch")
    
    train_sampler = DistributedSampler(
        train_dataset_filtered, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True, 
        drop_last=True
    )
    
    train_loader = DataLoader(
        train_dataset_filtered, 
        batch_size=BATCH_SIZE_PER_GPU, 
        sampler=train_sampler, 
        num_workers=2, 
        pin_memory=True
    )
    
    # Training loop variables
    ema = None
    ctx_idx = 0
    last_curv_warn_step = -100

    # Main training loop
    for epoch in range(MAX_STEPS // len(train_loader) + 1):
        train_sampler.set_epoch(epoch)
        if rank == 0 and epoch > 0: 
            print(f"Starting Epoch {epoch}")

        for i, batch in enumerate(train_loader):
            if step >= MAX_STEPS: 
                break

            # Dynamic DELTA / EPSILON adjustment
            if rank == 0:
                c_mean_val = core.C.mean().item()
                new_delta, new_epsilon = DELTA, EPSILON

                if c_mean_val < CURVATURE_MIN_THRESHOLD:
                    new_delta *= 1.05
                    log_only_meta_controller.log(f"Increasing curvature learning rate δ to {new_delta:.4f} at step {step}")
                
                if curvature_variance_high(core.C):
                    new_epsilon *= 0.95
                    log_only_meta_controller.log(f"Reducing curvature decay ε to {new_epsilon:.5f} at step {step}")

                dynamic_hyperparams[0] = new_delta
                dynamic_hyperparams[1] = new_epsilon
            
            # Broadcast DELTA and EPSILON
            dist.broadcast(dynamic_hyperparams, src=0)
            DELTA = dynamic_hyperparams[0].item()
            EPSILON = dynamic_hyperparams[1].item()

            # Context expansion
            if ctx_idx < len(CONTEXT_SCHEDULE) and step >= CONTEXT_SCHEDULE[ctx_idx][0]:
                target_ctx = CONTEXT_SCHEDULE[ctx_idx][1]
                if target_ctx > core.ctx_len:
                    core.expand_ctx(target_ctx, rank=rank)
                elif rank == 0:
                    print(f"Skipping context schedule entry {CONTEXT_SCHEDULE[ctx_idx]} as target length {target_ctx} is not greater than current {core.ctx_len}")
                ctx_idx += 1
            dist.barrier()

            input_ids = batch["input_ids"].to(current_device)
            labels = input_ids.clone()

            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                logits, local_block_activ_means = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
                    labels[:, 1:].reshape(-1)
                )
            
            current_loss_item = loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA-based G update and ACP curvature memory update
            if isinstance(optimizer, ARPOptimizer):
                # AllReduce local_block_activ_means
                global_block_activ_means = []
                for local_am_tensor in local_block_activ_means:
                    reduced_am_tensor = local_am_tensor.clone().detach()
                    dist.all_reduce(reduced_am_tensor, op=dist.ReduceOp.AVG)
                    global_block_activ_means.append(reduced_am_tensor)
                
                # Update G and C using global means
                with torch.no_grad():
                    # G update
                    for b_idx, globally_avg_act_stat_tensor in enumerate(global_block_activ_means):
                        core.G[b_idx] = core.G[b_idx] * (1 - MU) + ALPHA * globally_avg_act_stat_tensor * core.C[b_idx]
                    
                    # C update
                    loss_for_c_update = loss.clone().detach()
                    dist.all_reduce(loss_for_c_update, op=dist.ReduceOp.AVG)
                    loss_item_for_c = loss_for_c_update.item()

                    gamma = 2.0 - (loss_item_for_c / L_MAX)
                    base = max(L_MAX - loss_item_for_c, 0.0)
                    curvature_update = DELTA * (base ** gamma) - EPSILON * core.C
                    core.C += curvature_update
                    core.C = torch.clamp(core.C, min=0.1)

            # Rank 0: Logging, EMA, CPR, send to MetaLoop
            if rank == 0:
                ema = current_loss_item if ema is None else 0.99*ema + 0.01*current_loss_item
                
                # Send data to MetaLoop
                if step % (META_LOOP_CHECK_INTERVAL_STEPS // 4) == 0:
                    try:
                        meta_loop_q.put_nowait({'step': step, 'ema_loss': ema})
                    except queue.Full:
                        pass

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
                    if step % (LOG_INTERVAL * 10) == 0:
                        writer.add_histogram('Curvature/C_hist', core.C, step)
                    
                    # G logging
                    if step % (LOG_INTERVAL * 10) == 0:
                        writer.add_histogram("G/hist_all", core.G, step)

                    # Per-layer weight and gradient histograms (less frequently)
                    if step % (LOG_INTERVAL * 20) == 0:
                        for name, param in core.named_parameters():
                            if param.requires_grad:
                                writer.add_histogram(f'Weights/{name}', param.data, step)
                                if param.grad is not None:
                                    writer.add_histogram(f'Gradients_rank0/{name}', param.grad.data, step)
                    
                    writer.flush()
                    print(f"Epoch {epoch} | Step {step}/{MAX_STEPS} | Rank {rank} | Loss {current_loss_item:.4f} | EMA {ema:.4f} | CTX {core.ctx_len} | GradNorm {total_norm:.4f}")

                # CPR diagnostics
                if cpr_controller:
                    cpr_state = cpr_controller.update(current_loss_item)
                    if cpr_state == "TRIGGERED" and step % 500 == 0:
                        print(f"⚠️  CPR trigger at step {step} on Rank 0")
                        writer.add_scalar("CPR/trigger", 1, step)
                    elif cpr_state == "RESET":
                        print(f"🟢 CPR reset at step {step} on Rank 0")
                        writer.add_scalar("CPR/reset", 1, step)
                
                # Curvature mean high warning
                C_mean_threshold = 2.0
                if core.C.mean().item() > C_mean_threshold:
                    if step - last_curv_warn_step >= 100 * LOG_INTERVAL:
                        print(f"⚠️ Rank 0: Curvature mean high ({core.C.mean().item():.2f}) at step {step}")
                        last_curv_warn_step = step

                # Checkpointing and generation
                if step > 0 and step % CKPT_INTERVAL == 0:
                    fn = CKPT_DIR / f"gpt2_ddp_win_step{step}_{int(time.time())}.pth"
                    save_obj = {
                        "model_state_dict": core.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "step": step,
                        "ctx_len": core.ctx_len,
                        "ALPHA": optimizer.alpha if isinstance(optimizer, ARPOptimizer) else ALPHA,
                        "MU": optimizer.mu if isinstance(optimizer, ARPOptimizer) else MU,
                        "DELTA": DELTA,
                        "EPSILON": EPSILON
                    }
                    torch.save(save_obj, fn)
                    old_ckpts = sorted(CKPT_DIR.glob("gpt2_ddp_win_step*.pth"))[:-MAX_BACKUPS]
                    for f_old in old_ckpts: f_old.unlink(missing_ok=True)
                    print(f"💾 Checkpoint saved @ {step} → {fn}")
                    
                    # Generate sample text
                    sample_text = core.generate(GEN_PROMPT, temp=0.8, top_k=50)
                    writer.add_text("Sample Generation", sample_text, step)
                    print(f"Sample @ {step}: {sample_text}")

            # Synchronize optimizer parameters
            if isinstance(optimizer, ARPOptimizer):
                arp_params_to_sync = torch.tensor(
                    [optimizer.alpha, optimizer.mu] if isinstance(optimizer, ARPOptimizer) else [-1.0, -1.0], 
                    dtype=torch.float64, 
                    device=current_device
                )
                
                dist.broadcast(arp_params_to_sync, src=0)
                
                if rank != 0 and isinstance(optimizer, ARPOptimizer):
                    optimizer.alpha = arp_params_to_sync[0].item()
                    optimizer.mu = arp_params_to_sync[1].item()

            step += 1
            if step >= MAX_STEPS: 
                break
        
        if step >= MAX_STEPS: 
            break

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Windows-friendly DDP training for RealignR')
    parser.add_argument('--nproc_per_node', type=int, default=2, help='Number of processes per node')
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'], 
                        help='DDP backend (gloo recommended for Windows)')
    args = parser.parse_args()
    
    # Set the multiprocessing start method to 'spawn' for Windows compatibility
    mp.set_start_method('spawn')
    
    # Get number of GPUs
    world_size = args.nproc_per_node
    if torch.cuda.is_available():
        world_size = min(world_size, torch.cuda.device_count())
    print(f"Starting {world_size} processes with {args.backend} backend")
    
    # Launch the processes
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
