# realignr_gpt_char.py — Char-level GPT-style training using ARP

import sys, os
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'optimizers')))
from arp_optimizer import ARPOptimizer  # now works from optimizers/ folder

# --- Config ---
BATCH_SIZE = 64
SEQ_LEN = 64
EPOCHS = 5000
LOG_INTERVAL = 100
LR = 1e-3
ALPHA = 0.01
MU = 0.001
LOG_DIR = "runs/realignr_gpt_char"
CHECKPOINT_INTERVAL = 1000

os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)

# --- Tiny text corpus ---
text = """the quick brown fox jumps over the lazy dog. the quick brown fox jumps again."""
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
decode = lambda t: ''.join([itos[int(i)] for i in t])
data = encode(text)

# --- Char-level Dataset ---
class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

dataset = CharDataset(data, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Tiny GPT-like model ---
class CharModel(nn.Module):
    def __init__(self, vocab_size, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim))
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.G = torch.zeros(dim)  # ARP memory vector
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

# --- Training loop ---
step = 0
for epoch in range(EPOCHS):
    for xb, yb in dataloader:
        logits, x_embed, g = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.G += ALPHA * x_embed.abs().mean((0,1)).detach() - MU * model.G

        if step % LOG_INTERVAL == 0:
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("G_mean", model.G.mean().item(), step)
            print(f"Step {step:04d} | Loss: {loss.item():.4f} | G_mean: {model.G.mean():.4f}")

        if step % CHECKPOINT_INTERVAL == 0 and step > 0:
            torch.save(model.state_dict(), f"gpt_char_step{step}.pth")

        step += 1

# --- Test generation ---
def generate(model, start, steps=100):
    model.eval()
    input = encode(start).unsqueeze(0)
    out = [c for c in start]
    with torch.no_grad():
        for _ in range(steps):
            logits, _, _ = model(input)
            next_id = logits[0, -1].argmax()
            out.append(itos[int(next_id)])
            input = torch.cat([input, next_id.unsqueeze(0).unsqueeze(0)], dim=1)
    return ''.join(out)

print("\nGenerated: ", generate(model, "the "))
