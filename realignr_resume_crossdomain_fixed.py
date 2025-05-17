# ------------------------------------------------------------
# realignr_resume_crossdomain.py — *fresh-start, long-haul version*
# ------------------------------------------------------------
#  • Starts from HuggingFace GPT-2 weights (no checkpoint)
#  • AdamW warm-up → ARP after N epochs
#  • Dynamic context schedule: 1 024 → 2 048 → 4 096 tokens
#  • Two-GPU DataParallel (Windows-friendly, NCCL-free)
#  • TensorBoard with purge_step=0, auto-rotating checkpoints
# ------------------------------------------------------------

# ── expose both 3090 Ti GPUs *before* torch import ───────────
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print("CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])

# ── std / third-party ------------------------------------------------
import time, glob, math, json
from pathlib import Path
from itertools import chain

import torch, torch.nn as nn, torch.nn.functional as F
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler # Added for AMP

# ── local utilities --------------------------------------------------
from optimizers.arp_optimizer import ARPOptimizer
from meta_controller import MetaController, CPRController
from action import ActionTracker

# ── CONSTANTS --------------------------------------------------------
CURVATURE_MIN_THRESHOLD = 0.15  # Example threshold for C_ij.mean()
DELTA = 0.01
EPSILON = 0.001
L_MAX = 10.0  # Set this to a suitable reference loss or path-length measure
BASE_DIR   = Path(__file__).resolve().parent
STEP_START = 0  # Start from scratch
LOG_DIR    = BASE_DIR / "runs" / f"cpr_reset_sanity_{int(time.time())}"
CKPT_DIR   = BASE_DIR / "checkpoints" # Checkpoints are in the checkpoints folder
RESUME_CKPT = None  # No checkpoint for a fresh run

MAX_STEPS  = 300_000
CKPT_INTERVAL = 2_000
LOG_INTERVAL  = 50
MAX_BACKUPS   = 8
PLATEAU_PATIENCE = 500 # For loss plateau detection
PLATEAU_SLOPE_THRESHOLD = -0.001 # Slope threshold for plateau

SEQ_LEN   = 1_024 # Keep for data processing functions
CTX_START = 1_024 # Start model with this context length (start small, expand later)
BATCH_SIZE = 4
CONTEXT_SCHEDULE = [
    (50_000, 2_048),   # Expand to 2k at step 50k
    (120_000, 4_096),  # Expand to 4k at step 120k
]
ALPHA, MU = 0.01, 0.001
# ARP_SWITCH_EPOCHS = 20 # No longer needed

GEN_PROMPT = "The meaning of life is"
GEN_LEN    = 50

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True) # Create checkpoint directory

# ── TOKENIZER / DATA HELPERS ----------------------------------------
# Uses SEQ_LEN for padding/chunking
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def encode_batch(text_list):
    # Uses SEQ_LEN
    enc = tokenizer(text_list, padding="max_length", truncation=True,
                    max_length=SEQ_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

def tokenize_and_chunk(ex):
    ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
    flat = list(chain.from_iterable(ids))
    # Uses SEQ_LEN
    return {"input_ids": [flat[i:i+SEQ_LEN] for i in range(0, len(flat), SEQ_LEN)]}

def get_packed_wikitext103(split="train[:2%]"):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    ds = ds.map(tokenize_and_chunk, batched=True, remove_columns=["text"])
    # Uses SEQ_LEN
    ds = ds.filter(lambda e: len(e["input_ids"]) == SEQ_LEN).with_format("torch")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=split.startswith("train"))

# ── MODEL ------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
    def forward(self, x, mask):
        # attention + residual + layernorm
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + a)
        # feed-forward + residual + layernorm
        x = self.ln2(x + self.ff(x))
        # cache output for EMA-based G update
        self.last_out = x.detach()
        return x

class RealignRGPT2(nn.Module):
    # Accept ctx_len explicitly
    def __init__(self, vocab, ctx_len):
        super().__init__()
        dim = 768
        self.ctx_len  = ctx_len # Use provided ctx_len
        self.tok_emb  = nn.Embedding(vocab, dim)
        # Initialize pos_emb with the provided ctx_len
        self.pos_emb  = nn.Parameter(torch.zeros(1, ctx_len, dim))
        self.blocks   = nn.ModuleList(Block(dim, 12) for _ in range(4))
        self.ln_f     = nn.LayerNorm(dim)
        self.head     = nn.Linear(dim, vocab, bias=False)

        # Initialize G as a tensor with dimensions [n_blocks, d_model]
        g_init = torch.zeros(len(self.blocks), dim)
        self.register_buffer("G", g_init)
        # Initialize C (curvature memory) as all ones
        c_init = torch.ones(len(self.blocks), dim)
        self.register_buffer("C", c_init)

    def expand_ctx(self, new_len):
        if new_len <= self.ctx_len: return
        old = self.pos_emb.data; new = torch.zeros(1, new_len, old.size(2), device=old.device)
        new[:, :self.ctx_len] = old; nn.init.trunc_normal_(new[:, self.ctx_len:], std=0.02)
        self.pos_emb, self.ctx_len = nn.Parameter(new), new_len
        print(f"🔁 context window → {new_len}")
    def forward(self, idx):
        B,T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T]
        mask = torch.triu(torch.ones(T,T,device=idx.device)*float('-inf'), 1)

        activ_mean = []                              # collect |x| mean per block
        for blk in self.blocks:
            x = blk(x,mask)
            activ_mean.append(x.abs().mean((0, 1)))  # (d_model,)

        return self.head(self.ln_f(x)), activ_mean   # return per-block stats

    @torch.no_grad()
    def generate(self, prompt, max_len=GEN_LEN, temp=1.0, top_k=0):
        self.eval(); ids = tokenizer.encode(prompt, return_tensors="pt").to(self.head.weight.device)
        for _ in range(max_len):
            logits,_ = self(ids[:, -self.ctx_len:]); logits = logits[:,-1,:]/temp
            if top_k:
                topv, topi = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('inf')); logits.scatter_(1, topi, topv)
            next_tok = torch.multinomial(torch.softmax(logits,-1), 1); ids = torch.cat([ids,next_tok],-1)
        return tokenizer.decode(ids[0], skip_special_tokens=True)

# ── BUILD MODEL ----------------------------------------------
"""Initialize model with the context length matching the checkpoint"""
base_model = RealignRGPT2(tokenizer.vocab_size, ctx_len=CTX_START).cuda()
model      = torch.nn.DataParallel(base_model, device_ids=[0,1]); core = model.module
print("DataParallel devices:", model.device_ids, "| torch sees", torch.cuda.device_count(), "GPUs")
print(f"Model initialized with context length: {core.ctx_len}")

# Debug: Check G shape
print(f"G shape: {core.G.shape}") # Should be torch.Size([4, 768])

# ── DEFINE OPTIMISER (Starting directly with ARP) --------------------------
optimizer = ARPOptimizer(model.parameters(), alpha=ALPHA, mu=MU)
print(f"🔀 Resuming with ARP (α={ALPHA}, μ={MU}) from step {STEP_START}")

# Initialize GradScaler for AMP
scaler = GradScaler()

# ── RESUME FROM CHECKPOINT ----------------------------------------------
step = STEP_START # Initialize step with STEP_START
print("🚀 Starting fresh training run from scratch.")
print("First token weights L2:", core.tok_emb.weight[0].norm().item())

# ── CONTROLLERS ------------------------------------------------------
cpr  = CPRController(epsilon=1e-3, reset_patience=500)
meta = MetaController(); action_tracker = ActionTracker(λ1=0.15)

# ── DATA LOADER ------------------------------------------------------
train_loader = get_packed_wikitext103(); loader_iter = iter(train_loader)

# ── LOGGER -----------------------------------------------------------
writer = SummaryWriter(str(LOG_DIR), purge_step=STEP_START) # Use STEP_START for purge_step
print(f"▶️  Training starts/resumes at step {step}, ctx={core.ctx_len}") # Use loaded/updated step
print(f"[TensorBoard] Log directory: {LOG_DIR}")
print(f"[TensorBoard] Run: tensorboard --logdir \"{LOG_DIR}\"")

# Define helper function for curvature variance checking
def curvature_variance_high(C):
    # Example: consider variance high if std/mean > 0.5
    c_std = C.std().item()
    c_mean = C.mean().item()
    return c_mean > 0 and (c_std / c_mean) > 0.5

# ── CHECKPOINT HELPER -----------------------------------------------
def save_ckpt(step:int):
    # Save checkpoints to CKPT_DIR with unique timestamped names
    fn = CKPT_DIR / f"gpt2_s0_step{step}_{int(time.time())}.pth"
    # Include scaler state if using AMP
    save_obj = {"model_state_dict": core.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step}
    if 'scaler' in globals() and scaler is not None:
        save_obj["scaler_state_dict"] = scaler.state_dict()
    torch.save(save_obj, fn)
    # Only delete old checkpoints in CKPT_DIR, not globally
    old = sorted(CKPT_DIR.glob("gpt2_s0_step*.pth"))[:-MAX_BACKUPS]
    for f in old: f.unlink(missing_ok=True)
    print(f"💾 checkpoint saved @ {step} → {fn}")

# ── TRAIN LOOP -------------------------------------------------------
ema = None; ctx_idx = 0 # step is now loaded or set above
# Remove the pre-loop context expansion logic as model starts with correct ctx_len
# The loop's dynamic expansion logic will handle future increases based on CONTEXT_SCHEDULE
print(f"Initial context index set to {ctx_idx} (no pre-expansion needed)")

last_curv_warn_step = -100  # For throttling curvature mean warnings
loss_history = [] # For plateau detection

# Helper function for loss slope calculation
def calculate_slope(loss_window):
    if not loss_window or len(loss_window) < 2:
        return 0 # Not enough data to calculate slope
    y = torch.tensor(loss_window, dtype=torch.float32)
    x = torch.arange(len(y), dtype=torch.float32)
    # Using simple linear regression (polyfit deg=1)
    # For PyTorch tensors, we can do it manually or use a library if available
    # Manual calculation for slope: (N * sum(xy) - sum(x)sum(y)) / (N * sum(x^2) - (sum(x))^2)
    N = len(x)
    sum_xy = torch.sum(x * y)
    sum_x = torch.sum(x)
    sum_y = torch.sum(y)
    sum_x_sq = torch.sum(x**2)
    
    denominator = N * sum_x_sq - sum_x**2
    if denominator == 0:
        return 0 # Avoid division by zero
    slope_val = (N * sum_xy - sum_x * sum_y) / denominator
    return slope_val.item()

while step < MAX_STEPS:
    # -- Curvature logging and anomaly detection --

    # Curvature memory C logging (global and per-block)
    writer.add_scalar('Curvature/C_mean', core.C.mean().item(), step)
    writer.add_scalar('Curvature/C_std', core.C.std().item(), step)
    writer.add_histogram('Curvature/C_hist', core.C, step)
    for k in range(core.C.size(0)):
        writer.add_scalar(f'Curvature/C_block{k}_mean', core.C[k].mean().item(), step)
        writer.add_scalar(f'Curvature/C_block{k}_std', core.C[k].std().item(), step)

    # Throttle "curvature mean high" warnings to reduce console spam
    C_mean_threshold = 2.0  # Example threshold for anomaly
    if core.C.mean().item() > C_mean_threshold:
        if step - last_curv_warn_step >= 100:
            print(f"⚠️ Curvature mean high at step {step}")
            last_curv_warn_step = step
    
    # --- Dynamic curvature learning rate and decay adjustment ---
    c_mean = core.C.mean().item()
    if c_mean < CURVATURE_MIN_THRESHOLD:
        DELTA *= 1.05
        meta.log(f"Increasing curvature learning rate δ to {DELTA:.4f} at step {step}")
    
    # Log DELTA and EPSILON hyperparameters
    writer.add_scalar('Curvature/DELTA', DELTA, step)
    writer.add_scalar('Curvature/EPSILON', EPSILON, step)
    
    if curvature_variance_high(core.C):
        EPSILON *= 0.95
        meta.log(f"Reducing curvature decay ε to {EPSILON:.5f} at step {step}")
    
    # dynamic ctx expansion (handles increases > CTX_START)
    if ctx_idx < len(CONTEXT_SCHEDULE) and step >= CONTEXT_SCHEDULE[ctx_idx][0]:
        target_ctx = CONTEXT_SCHEDULE[ctx_idx][1]
        if target_ctx > core.ctx_len:
            print(f"🔁 Expanding context window: {core.ctx_len} → {target_ctx} at step {step}")
            core.expand_ctx(target_ctx)
        else:
            print(f"Skipping context schedule entry {CONTEXT_SCHEDULE[ctx_idx]} as target length is not greater than current {core.ctx_len}")
        ctx_idx += 1

    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(train_loader); batch = next(loader_iter)

    input_ids = batch["input_ids"].cuda(); labels = input_ids.clone()
    # -- loss & optimiser -------------------------------
    optimizer.zero_grad(set_to_none=True) # Recommended for performance with AMP

    with autocast(): # AMP: Casts operations to mixed precision
        logits, activ_mean = model(input_ids) # Capture activ_mean
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
            labels[:, 1:].reshape(-1)
        )
    
    # AMP: Scales loss and calls backward
    scaler.scale(loss).backward()
    
    # AMP: Unscales gradients and clips grad norm
    scaler.unscale_(optimizer) # Unscale before clipping
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # AMP: Optimizer step
    scaler.step(optimizer)
    
    # AMP: Updates the scale for next iteration
    scaler.update()
    
    # -- EMA-based G update (block-wise exponential moving average, with curvature) ----------
    if isinstance(optimizer, ARPOptimizer):
        with torch.no_grad():
            # activ_mean is a list of per-block activation stats (possibly stacked across GPUs)
            for b, act_stat in enumerate(activ_mean):
                # act_stat may be shape (num_replicas, d_model) or (d_model,)
                # compute scalar mean over all elements
                act_abs_mean = act_stat.mean()
                # EMA update: decay by (1-MU), then add ALPHA * act_abs_mean * C[b]
                core.G[b] = core.G[b] * (1 - MU) + ALPHA * act_abs_mean * core.C[b]
        
        # -- ACP curvature memory update (after G update)
        with torch.no_grad():
            gamma = 2.0 - (loss.item() / L_MAX)
            base = max(L_MAX - loss.item(), 0.0)
            curvature_update = DELTA * (base ** gamma) - EPSILON * core.C
            core.C += curvature_update
            core.C = torch.clamp(core.C, min=0.1)  # keep C positive
    
    # -- CPR diagnostics --------------------------------
    state = cpr.update(loss.item())
    if state == "TRIGGERED" and step % 500 == 0:
        print(f"⚠️  CPR trigger at step {step}")
        writer.add_scalar("CPR/trigger", 1, step)
    elif state == "RESET":
        print(f"🟢 CPR reset at step {step}")
        writer.add_scalar("CPR/reset", 1, step)

    # -- EMA & logging ----------------------------------
    current_loss_item = loss.item()
    ema = current_loss_item if ema is None else 0.99*ema + 0.01*current_loss_item
    loss_history.append(current_loss_item)
    if len(loss_history) > PLATEAU_PATIENCE * 2: # Keep history manageable
        loss_history.pop(0)

    if step % LOG_INTERVAL == 0:
        writer.add_scalar("Loss/train", current_loss_item, step)
        writer.add_scalar("Loss/train_smooth", ema, step)
        writer.add_scalar("Gradients/total_norm", total_norm.item(), step)
        writer.add_scalar("AMP/grad_scale", scaler.get_scale(), step)
        writer.add_scalar("ctx_len", core.ctx_len, step)
        
        # Per-layer weight and gradient histograms
        if step % 1000 == 0: # Log histograms less frequently
            for name, param in core.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'Weights/{name}', param.data, step)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad.data, step)
        
        writer.flush()  # Explicit flush to ensure event file is updated
        print(f"step {step} | loss {current_loss_item:.4f} | ema {ema:.4f} | ctx {core.ctx_len} | grad_norm {total_norm:.4f}")

    # -- Learning rate / optimizer parameter scheduler based on loss plateau --
    if len(loss_history) >= PLATEAU_PATIENCE and step % PLATEAU_PATIENCE == 0: # Check every PLATEAU_PATIENCE steps
        # Consider the most recent PLATEAU_PATIENCE points for slope calculation
        slope = calculate_slope(loss_history[-PLATEAU_PATIENCE:])
        writer.add_scalar("Scheduler/loss_slope", slope, step)
        if slope > PLATEAU_SLOPE_THRESHOLD:  # Plateau detected (slope is not sufficiently negative)
            if isinstance(optimizer, ARPOptimizer):
                old_alpha, old_mu = optimizer.alpha, optimizer.mu
                optimizer.alpha *= 0.7  # Decay alpha
                optimizer.mu *= 1.05    # Bump mu
                # Ensure mu doesn't grow excessively, e.g. cap it or make bump smaller if already large
                # optimizer.mu = min(optimizer.mu, SOME_MAX_MU_VALUE) 
                meta.log(f"Plateau detected at step {step}. Slope: {slope:.5f}. ARPOptimizer: α {old_alpha:.4f}→{optimizer.alpha:.4f}, μ {old_mu:.4f}→{optimizer.mu:.4f}")
                writer.add_text("Scheduler/plateau_event", f"Plateau: α→{optimizer.alpha:.4f}, μ→{optimizer.mu:.4f}", step)
                writer.add_scalar("Scheduler/ARPO_alpha", optimizer.alpha, step)
                writer.add_scalar("Scheduler/ARPO_mu", optimizer.mu, step)
            else: # Fallback for other optimizers if needed, or adjust DELTA/EPSILON
                # Example: DELTA *= 0.9; EPSILON *= 0.9
                meta.log(f"Plateau detected at step {step}. Slope: {slope:.5f}. (Non-ARP optimizer, no α/μ adjustment)")
                writer.add_text("Scheduler/plateau_event", f"Plateau detected. Slope: {slope:.5f}", step)
            
            loss_history.clear() # Reset history after adjustment

    # -- TensorBoard G logging (optional) ----------------
    if step % 1000 == 0:
        writer.add_histogram("G/hist_all", core.G, step)
    for k in range(core.G.size(0)):
        writer.add_scalar(f"G_block{k}/mean", core.G[k].mean().item(), step)
        writer.add_scalar(f"G_block{k}/std", core.G[k].std().item(), step)

    # -- checkpoints ------------------------------------
    if step and step % CKPT_INTERVAL == 0:
        save_ckpt(step)
        writer.add_text("Sample", core.generate(GEN_PROMPT, temp=0.8, top_k=50), step)

    step += 1

print("🎉 training loop finished")
print("Logging to:", LOG_DIR)
writer.flush()
writer.close()
print("✅ TensorBoard logs flushed and closed")
