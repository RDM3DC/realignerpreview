# === api_server.py ===
"""FastAPI control surface for RealignR trainer.
Run with:  uvicorn api_server:app --port 8080  (or via forever script).
Exposes:
  GET /metrics              -> live trainer metrics
  POST /param {alpha,mu}    -> live optimiser tweak
  POST /dataset {name}      -> queue dataset switch
  POST /checkpoint/save {tag}
  POST /checkpoint/restore {path, reset}
"""
from fastapi import FastAPI
from pydantic import BaseModel
import queue, time, pathlib, torch, shutil

app = FastAPI()
cmd_q: queue.Queue = queue.Queue()          # consumed by training loop
metrics_store = {
    "step": 0,
    "loss": None,
    "loss_smooth": None,
    "g_mean": None,
    "drift": None,
    "dataset": "unknown",
    "ctx_len": 128,
}

class ParamReq(BaseModel):
    alpha: float | None = None
    mu: float | None = None

class DataReq(BaseModel):
    name: str

class CheckpointReq(BaseModel):
    tag: str | None = None

class RestoreReq(BaseModel):
    path: str
    reset: bool = False

@app.get("/metrics")
def get_metrics():
    return metrics_store

@app.post("/param")
def set_param(req: ParamReq):
    cmd_q.put(("param", req.dict(exclude_none=True)))
    return {"status": "queued"}

@app.post("/dataset")
def switch_dataset(req: DataReq):
    cmd_q.put(("dataset", req.name))
    return {"status": "queued"}

@app.post("/checkpoint/save")
def save_ckpt(req: CheckpointReq):
    cmd_q.put(("ckpt_save", req.tag or time.strftime("%Y%m%d_%H%M%S")))
    return {"status": "queued"}

@app.post("/checkpoint/restore")
def restore_ckpt(req: RestoreReq):
    cmd_q.put(("ckpt_restore", req.dict()))
    return {"status": "queued"}

# === trainer_hook.py ===
"""Utilities to drop into realignr_train.py"""
from pathlib import Path

def safe_save(model, path: str):
    l2 = model.transformer.wte.weight[0].norm().item()
    if l2 < 1.8:
        print(f"🚨  REFUSE save, emb_L2={l2:.2f} looks random")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾  Saved {path} | emb_L2={l2:.2f}")


def trainer_loop_step_hook(step, model, optimizer, cmd_q, current_loader, loaders_dict):
    """Call once per *epoch* or every N steps to process API commands."""
    processed = 0
    while not cmd_q.empty():
        cmd, payload = cmd_q.get()
        processed += 1
        if cmd == "param":
            for g in optimizer.param_groups:
                if "alpha" in payload: g["alpha"] = payload["alpha"]
                if "mu"   in payload: g["mu"]   = payload["mu"]
            print("⚙️  Updated alpha/mu via API", payload)
        elif cmd == "dataset":
            if payload in loaders_dict:
                current_loader.dataset = loaders_dict[payload]
                print(f"🔄  Dataset switched to {payload}")
        elif cmd == "ckpt_save":
            safe_save(model, f"checkpoints/api_{payload}.pth")
        elif cmd == "ckpt_restore":
            state = torch.load(payload["path"], map_location="cpu")
            model.load_state_dict(state, strict=True)
            if payload.get("reset"):
                model.global_step = 0
            print(f"🔁  Restored {payload['path']}")
    if processed:
        print(f"API: processed {processed} queued command(s)")

# Remove module-level watcher_gpt code to avoid HTTP calls on import
