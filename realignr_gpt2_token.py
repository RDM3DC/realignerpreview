# realignr_gpt2_token.py — Token-level GPT-2 with RealignR + Dataset Switching

import os, time, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from optimizers.arp_optimizer import ARPOptimizer
from meta_controller import MetaController

# --- Config ---
MODEL_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 4
SEQ_LEN = 128
BATCH_SIZE = 8
MAX_STEPS = 50000
ALPHA = 0.0025
MU = 0.001
LOG_INTERVAL = 50
CHECKPOINT_INTERVAL = 1000
LOG_DIR = "runs/realignr_gpt2_token"
CONTROL_FILE = "realignr_control.json"
PHASE_FILE = "realignr_phase.json"
GPT_LOG_FILE = "gpt_optimizer_feedback.log"
GEN_PROMPT = "The meaning of life is"
GEN_LEN = 50

# Dataset order used during training
SCHEDULE = {0: "wikitext", 50000: "tinystories"}

os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)

# --- Tokenizer ---
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# --- Dataset Loader ---
def encode_batch(batch):
    out = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=SEQ_LEN)
    return {"input_ids": torch.tensor(out["input_ids"])}

def get_dataset(name):
    if name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif name == "tinystories":
        dataset = load_dataset("roneneldan/TinyStories", split="train[:5%]")
    else:
        raise ValueError(f"Unknown dataset: {name}")
    dataset = dataset.filter(lambda e: len(e['text'].strip()) > 0)
    dataset = dataset.map(encode_batch, batched=True, remove_columns=["text"])
    # ensure PyTorch tensors on load
    dataset.set_format(type="torch", columns=["input_ids"])
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Dataset Schedule ---
# "SCHEDULE" is defined above as a simple dictionary mapping step numbers to
# dataset names. Convert it to the internal format expected below.
schedule = SCHEDULE

# --- Model ---
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

class RealignRGPT2(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, MODEL_DIM)
        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, MODEL_DIM))
        self.blocks = nn.ModuleList([Block(MODEL_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(MODEL_DIM)
        self.head = nn.Linear(MODEL_DIM, vocab_size)
        self.G = torch.zeros(MODEL_DIM).cuda()
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        attn_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(x.device)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.head(self.ln_f(x)), x

    def generate(self, prompt, max_len=GEN_LEN):
        self.eval()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.head.weight.device)
        for _ in range(max_len):
            logits, _ = self(input_ids[:, -SEQ_LEN:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return tokenizer.decode(input_ids[0])

# --- Init ---
model = RealignRGPT2(tokenizer.vocab_size).cuda()
optimizer = ARPOptimizer(model.parameters(), alpha=ALPHA, mu=MU)
meta = MetaController()
step = 0
current_loader = get_dataset(schedule[0])
dataset_name = schedule[0]

# --- Training Loop ---
loader_iter = iter(current_loader)
for step in range(MAX_STEPS):
    if step in schedule and schedule[step] != dataset_name:
        dataset_name = schedule[step]
        current_loader = get_dataset(dataset_name)
        loader_iter = iter(current_loader)
        writer.add_text("DatasetSwitch", f"Step {step}: switched to {dataset_name}", step)
        print(f"🔁 Dataset switch at step {step} → {dataset_name}")

    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(current_loader)
        batch = next(loader_iter)

    input_ids = batch["input_ids"].to(torch.device("cuda"))
    logits, x_embed = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))

    optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.G += (ALPHA * x_embed.abs().mean((0,1)).detach() - MU * model.G).to(model.G.device)

    if step % LOG_INTERVAL == 0:
        g_mean = model.G.mean().item()
        writer.add_scalar("Loss/train", loss.item(), step)
        writer.add_scalar("G_mean", g_mean, step)
        writer.add_text("Dataset", dataset_name, step)
        print(f"Step {step:05d} | Loss: {loss:.4f} | G_mean: {g_mean:.4f} | Dataset: {dataset_name}")
        meta.update(step, loss.item(), g_mean, cpr_trigger=0)

        if os.path.exists(PHASE_FILE):
            with open(PHASE_FILE, "r") as f:
                phase_data = json.load(f)
                writer.add_text("MetaPhase", f"{phase_data.get('phase')} — {phase_data.get('reason')}", step)

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

# --- Sample Generation ---
print("\n🧠 Sample output from RealignR-GPT2 after training:")
print(model.generate(GEN_PROMPT, GEN_LEN))
print("\n[Training complete — checkpoint saved]")
