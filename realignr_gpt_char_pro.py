# realignr_gpt_char_pro.py — Pro mode with slope-trend-aware GPT tuning

import os, random, time, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from optimizers.arp_optimizer import ARPOptimizer

# --- Config ---
BATCH_SIZE = 64
SEQ_LEN = 64
EPOCHS = 5000
LOG_INTERVAL = 100
LR = 1e-3
ALPHA = 0.01
MU = 0.001
LOG_DIR = "runs/realignr_gpt_char_pro"
CONTROL_FILE = "realignr_control.json"
GPT_LOG_FILE = "gpt_optimizer_feedback.log"
CHECKPOINT_INTERVAL = 1000
TEMPERATURE = 0.8
TOP_K = 5
CPR_THRESHOLD = 0.002  # if G_mean and loss slope stall
SLOPE_WINDOW = 5

os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)

# --- Load dataset (Shakespeare or default) ---
try:
    with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
        print("📖 Loaded TinyShakespeare.txt")
except FileNotFoundError:
    text = "the quick brown fox jumps over the lazy dog. the quick brown fox jumps again."
    print("📘 Using default demo corpus")

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
decode = lambda t: ''.join([itos[int(i)] for i in t])
data = encode(text)

# --- Dataset ---
class CharDataset(Dataset):
    def __init__(self, data, seq_len): self.data, self.seq_len = data, seq_len
    def __len__(self): return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

dataset = CharDataset(data, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
class CharModel(nn.Module):
    def __init__(self, vocab_size, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim))
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.G = torch.zeros(dim)
        self.out = nn.Linear(dim, vocab_size)
    def forward(self, x):
        x = self.emb(x)
        g = self.G.to(x.device)
        a,_ = self.attn(x, x, x)
        x = self.ln1(x + a)
        f = self.ffn(x) + g.unsqueeze(0).unsqueeze(0)
        x = self.ln2(x + f)
        return self.out(x), x, g

# --- Initialize ---
model = CharModel(vocab_size)
optimizer = ARPOptimizer(model.parameters(), alpha=ALPHA, mu=MU)
step = 0
loss_window = []

# --- Training Loop ---
for epoch in range(EPOCHS):
    for xb, yb in dataloader:
        logits, x_embed, g = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.G += ALPHA * x_embed.abs().mean((0,1)).detach() - MU * model.G

        if step % LOG_INTERVAL == 0:
            g_mean = model.G.mean().item()
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("G_mean", g_mean, step)
            print(f"Step {step:04d} | Loss: {loss.item():.4f} | G_mean: {g_mean:.4f}")

            loss_window.append(loss.item())
            if len(loss_window) > SLOPE_WINDOW:
                loss_window.pop(0)
                slope = (loss_window[-1] - loss_window[0]) / SLOPE_WINDOW
                writer.add_scalar("loss_slope", slope, step)
                print(f"📈 Loss slope over {SLOPE_WINDOW}: {slope:.6f}")

                if abs(slope) < CPR_THRESHOLD and g_mean < 0.05:
                    print("💥 CPR EVENT: Resetting G (slope stalled)")
                    model.G.mul_(0.2)
                    writer.add_scalar("CPR_trigger", 1.0, step)

            # --- GPT Control Update ---
            if os.path.exists(CONTROL_FILE):
                try:
                    with open(CONTROL_FILE, "r", encoding="utf-8") as f:
                        ctrl = json.load(f)
                    alpha_new = ctrl.get("alpha", ALPHA)
                    mu_new = ctrl.get("mu", MU)
                    for group in optimizer.param_groups:
                        group['alpha'] = alpha_new
                        group['mu'] = mu_new
                    writer.add_scalar("alpha_live", alpha_new, step)
                    writer.add_scalar("mu_live", mu_new, step)
                    print(f"📡 GPT UPDATE → alpha: {alpha_new:.5f} | mu: {mu_new:.5f}")
                    with open(GPT_LOG_FILE, "a", encoding="utf-8") as log:
                        log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] step {step} → {ctrl}\n")
                except Exception as e:
                    print(f"⚠️ GPT control read error: {e}")

        if step % CHECKPOINT_INTERVAL == 0 and step > 0:
            torch.save(model.state_dict(), f"gpt_char_pro_step{step}.pth")

        step += 1

# --- Generation ---
def sample(logits, temperature=1.0, top_k=None):
    logits = logits / temperature
    if top_k:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        probs = F.softmax(topk_vals, dim=-1)
        idx = torch.multinomial(probs, 1)
        return topk_idx.gather(0, idx).item()
    else:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

def generate(model, start, steps=100):
    model.eval()
    input = encode(start).unsqueeze(0)
    out = [c for c in start]
    with torch.no_grad():
        for _ in range(steps):
            logits, _, _ = model(input)
            next_id = sample(logits[0, -1], temperature=TEMPERATURE, top_k=TOP_K)
            out.append(itos[next_id])
            input = torch.cat([input, torch.tensor([[next_id]])], dim=1)
    return ''.join(out)

print("\nGenerated with sampling:")
print(generate(model, "the "))
