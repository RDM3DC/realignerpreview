import os, time, json, torch, torch.nn as nn, torch.nn.functional as F
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "optimizers"))

# realignr_gpt2_token.py — Token-level GPT-2 with RealignR

from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from arp_optimizer import ARPOptimizer
from meta_controller import MetaController

# --- Config ---
MODEL_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 4
SEQ_LEN = 128
BATCH_SIZE = 8
MAX_STEPS = 20000
ALPHA = 0.0025
MU = 0.001
LOG_INTERVAL = 50
CHECKPOINT_INTERVAL = 1000
LOG_DIR = "runs/realignr_gpt2_token"
CONTROL_FILE = "realignr_control.json"
PHASE_FILE = "realignr_phase.json"
GPT_LOG_FILE = "gpt_optimizer_feedback.log"

os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)

# --- Tokenizer and Dataset ---
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def encode_batch(batch):
    encodings = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
    return {"input_ids": encodings["input_ids"]}

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset = dataset.filter(lambda e: len(e['text'].strip()) > 0)
dataset = dataset.map(encode_batch, batched=True, remove_columns=["text"])
# initial DataLoader for training
_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=lambda batch: {"input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch])}
)
dataloader = _dataloader

# curriculum dataset schedule: switch datasets at specified step thresholds
DATASET_SCHEDULE = {0: "wikitext", 50000: "tinystories"}
current_dataset = None

def switch_dataset(step):
    global dataloader, current_dataset
    name = None
    for threshold in sorted(DATASET_SCHEDULE):
        if step >= threshold:
            name = DATASET_SCHEDULE[threshold]
    if name != current_dataset:
        current_dataset = name
        print(f"[DATA] Switching to dataset '{name}' at step {step}")
        # load raw dataset
        if name == "wikitext":
            ds0 = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        else:
            ds0 = load_dataset(name, split="train")
        ds0 = ds0.filter(lambda e: len(e['text'].strip()) > 0)
        ds0 = ds0.map(encode_batch, batched=True, remove_columns=["text"])
        # recreate DataLoader
        dataloader = torch.utils.data.DataLoader(
            ds0, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=lambda batch: {"input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch])}
        )

# --- GPT Block ---
class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, x, mask):
        a,_ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + a)
        f = self.ff(x)
        x = self.ln2(x + f)
        return x

# --- RealignR GPT-2 Mini Model ---
class RealignRGPT2(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, MODEL_DIM)
        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, MODEL_DIM))
        self.blocks = nn.ModuleList([Block(MODEL_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(MODEL_DIM)
        self.head = nn.Linear(MODEL_DIM, vocab_size)
        # register G as buffer so it moves with model devices
        self.register_buffer('G', torch.zeros(MODEL_DIM))
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        attn_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(x.device)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.head(self.ln_f(x)), x

# --- Init ---
model = RealignRGPT2(tokenizer.vocab_size).cuda()
optimizer = ARPOptimizer(model.parameters(), alpha=ALPHA, mu=MU)
meta = MetaController()
step = 0

# --- Training ---
for epoch in range(MAX_STEPS):
    for batch in dataloader:
        switch_dataset(step)
        input_ids = batch["input_ids"].to(torch.device("cuda"))
        logits, x_embed = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.G += ALPHA * x_embed.abs().mean((0,1)).detach() - MU * model.G

        if step % LOG_INTERVAL == 0:
            g_mean = model.G.mean().item()
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("G_mean", g_mean, step)
            print(f"Step {step:04d} | Loss: {loss:.4f} | G_mean: {g_mean:.4f}")
            meta.update(step, loss.item(), g_mean, cpr_trigger=0)

            if os.path.exists(PHASE_FILE):
                with open(PHASE_FILE, "r") as f:
                    phase_data = json.load(f)
                    writer.add_text("MetaPhase", f"{phase_data.get('phase')} — {phase_data.get('reason')}", step)
                    # soft decay alpha during stable phase toward equilibrium alpha_target = mu * 1.9184
                    phase = phase_data.get('phase')
                    if phase == "stable":
                        for group in optimizer.param_groups:
                            alpha_curr = group.get('alpha', ALPHA)
                            mu_curr = group.get('mu', MU)
                            alpha_target = mu_curr * 1.9184
                            new_alpha = (alpha_curr + alpha_target) / 2
                            group['alpha'] = new_alpha
                        writer.add_scalar("alpha_stable_decay", new_alpha, step)
                        print(f"[META] Decayed alpha in stable phase to {new_alpha:.5f}")

            if os.path.exists(CONTROL_FILE):
                try:
                    with open(CONTROL_FILE, "r") as f:
                        ctrl = json.load(f)
                        alpha_new = ctrl.get("alpha", ALPHA)
                        mu_new = ctrl.get("mu", MU)
                        for group in optimizer.param_groups:
                            group['alpha'] = alpha_new
                            group['mu'] = mu_new
                        writer.add_scalar("alpha_live", alpha_new, step)
                        writer.add_scalar("mu_live", mu_new, step)
                        print(f"📡 GPT UPDATE -> alpha: {alpha_new:.5f} | mu: {mu_new:.5f}")
                    with open(GPT_LOG_FILE, "a", encoding="utf-8") as log:
                        log.write(f"[step {step}] -> {ctrl}\n")
                except Exception as e:
                    print(f"⚠️ GPT control read error: {e}")

        if step % CHECKPOINT_INTERVAL == 0 and step > 0:
            torch.save(model.state_dict(), f"gpt2_realignr_step{step}.pth")

        step += 1
        if step >= MAX_STEPS:
            break

print("\n[Training complete — checkpoint saved]")