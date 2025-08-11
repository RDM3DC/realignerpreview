"""Tiny Transformer benchmark on a subset of WikiText-103."""
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from torch.optim import AdamW, Adagrad

from optim.realignr import RealignR
from utils.plateau import plateau_detected


def get_loaders(batch_size: int = 8, seq_len: int = 128) -> tuple[DataLoader, DataLoader]:
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def encode(batch):
        tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=seq_len)
        return {"input_ids": tokens["input_ids"]}

    tokenized = dataset.map(encode, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids"])

    train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized["validation"], batch_size=batch_size)
    return train_loader, val_loader


def evaluate(model: GPT2LMHeadModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            total_loss += loss.item() * inputs.numel()
            total_tokens += inputs.numel()
    return math.exp(total_loss / total_tokens)


def run(track: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders()

    config = GPT2Config(n_layer=12, n_head=12, n_embd=768)
    model = GPT2LMHeadModel(config).to(device)

    steps = 100_000
    eval_interval = 1000
    switch_step = 50_000

    if track == "adamw":
        opt = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        switch_done = True
    elif track == "adagrad":
        opt = Adagrad(model.parameters(), lr=2e-4)
        switch_done = True
    else:
        opt = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        switch_done = False

    val_hist: List[float] = []
    grad_var_hist: List[float] = []

    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", f"tiny_transformer_{track}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "val_ppl", "grad_var", "snr"])

        step = 0
        grad_norm_sum = 0.0
        grad_norm_sq_sum = 0.0
        count = 0
        while step < steps:
            for batch in train_loader:
                model.train()
                inputs = batch["input_ids"].to(device)
                opt.zero_grad()
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                loss.backward()

                g_vec = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
                g_norm = g_vec.norm().item()
                grad_norm_sum += g_norm
                grad_norm_sq_sum += g_norm ** 2
                count += 1

                opt.step()
                step += 1

                if step % eval_interval == 0:
                    grad_mean = grad_norm_sum / count
                    grad_var = grad_norm_sq_sum / count - grad_mean ** 2
                    snr = grad_mean ** 2 / (grad_var + 1e-8)
                    val_ppl = evaluate(model, val_loader, device)
                    val_hist.append(val_ppl)
                    grad_var_hist.append(grad_var)
                    writer.writerow([step, val_ppl, grad_var, snr])

                    if not switch_done and (plateau_detected(val_hist, grad_var_hist) or step >= switch_step):
                        lr_final = opt.param_groups[0]["lr"]
                        if track == "adamw_to_adagrad":
                            opt = Adagrad(model.parameters(), lr=lr_final)
                        elif track == "adamw_to_realignr":
                            opt = RealignR(model.parameters(), lr=lr_final * 0.7, mu=0.1, alpha=1.0)
                        elif track == "adamw_to_realignr_cma":
                            opt = RealignR(
                                model.parameters(),
                                lr=lr_final * 0.7,
                                mu=0.1,
                                alpha=1.0,
                                cma_xi=0.1,
                                cma_beta=0.05,
                            )
                        switch_done = True

                if step >= steps:
                    break

    print(f"Log saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Transformer WikiText benchmark")
    parser.add_argument(
        "track",
        type=str,
        default="adamw",
        choices=[
            "adamw",
            "adagrad",
            "adamw_to_adagrad",
            "adamw_to_realignr",
            "adamw_to_realignr_cma",
        ],
        help="Optimization track to run",
    )
    args = parser.parse_args()
    run(args.track)
