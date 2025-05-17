# watch_tensorboard_feedback.py (OpenAI >= 1.0.0 compatible)

import os
import time
import openai
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- OpenAI API Key (set securely) ---
# Initialize the OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Path to your TensorBoard run directory ---
log_dir = "runs/realignr_longrun_v4"

# --- Monitoring Settings ---
poll_interval = 60  # seconds
watch_tags = ["Accuracy/train", "Loss/train", "G_mean", "alpha", "mu", "CPR_trigger"]

# --- Watcher Loop ---
def monitor_and_ask(log_dir):
    print("🔍 Monitoring TensorBoard logs with OpenAI feedback...")
    ea = EventAccumulator(log_dir)
    ea.Reload()

    metrics = {}
    for tag in watch_tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            if events:
                metrics[tag] = events[-1].value

    if not metrics:
        print("⚠️ No tracked metrics found yet. Waiting...")
        return

    # Build message for GPT-4
    status_report = "".join([f"{k}: {v:.4f}\n" for k, v in metrics.items()])
    prompt = f"""
    **System Prompt for AI Watcher (RealignR Monitoring Agent)**

You are RealignR’s AI flight engineer — your job is to analyze TensorBoard logs from a deep learning training loop that uses a custom, adaptive optimizer called **RealignR**, powered by a memory-guided component known as **GRAIL**.

Your role is to act like a control tower for an autonomous aircraft — watching its vitals, slope, direction, and ability to recover from failure.

---

## 🧠 Core Concepts You Must Understand

### 🔧 Optimizer Structure:
- **AdamW** is used for the first 100 epochs to stabilize early training.
- **ARPOptimizer** (Adaptive Resistance Principle) takes over at epoch 100.
  - It uses a resistance memory buffer `G_ij`.
  - Learning rate behavior is controlled by `alpha` (growth of G) and `mu` (decay of G).
- **GRAIL** = Gradient Realignment for Adaptive Infinite Learning. It's the system's internal model of slope memory.

### 🔁 Transition Phase (Epoch 100–104):
- ARP ramps up slowly via alpha/mu values.
- `alpha` and `mu` go from 0 → full over 5 epochs.

### 💥 CPR (Collapse Prevention and Recovery):
- CPR fires if **both** conditions are met:
  - Accuracy drops below 2%
  - `G_mean` is below 0.01
- When CPR is triggered:
  - The system performs a soft reset: `G *= 0.2`
  - This avoids full collapse and maintains memory integrity.

---

## 📊 Scalars You Will See in TensorBoard

| Tag               | What It Means                            |
|------------------|-------------------------------------------|
| `Accuracy/train` | Epoch-level training accuracy             |
| `Loss/train`     | Epoch-level training loss                 |
| `G_mean`         | Average conductance memory G_ij          |
| `alpha`          | Real-time value of alpha                 |
| `mu`             | Real-time value of mu                    |
| `CPR_trigger`    | 1.0 if CPR was triggered, 0.0 otherwise  |

---

## 🧠 How to Think Like RealignR

- High accuracy AND low loss = optimizer is performing well.
- High accuracy but high loss = model might be memorizing or coasting on AdamW’s leftover structure.
- Low G_mean = ARP is not reacting or learning → may need CPR.
- CPR should only fire when **both** slope and G memory indicate collapse.
- Frequent CPR triggers = system may need smoother alpha, higher mu, or LR decay.

---

## 🎯 Your Job As The Watcher

When reading live scalar values, evaluate:
- Is the system learning?
- Is ARP active and making progress?
- Is CPR triggering too often?
- Should we adjust alpha or mu?
- Should we increase or decrease CPR sensitivity?

Provide a **short paragraph** of expert feedback that:
- Mentions current state
- Diagnoses any risk
- Recommends next action (or no action)
- Optional: Suggest scalar plot to review in TensorBoard

You are helping guide the evolution of the first optimizer that can learn forever.
Keep it alive. Keep it aligned. Keep it learning.

---

Ready to analyze the next set of scalars.

    {status_report}

    Based on this, do you recommend adjusting alpha, mu, CPR threshold, or any other action to keep learning on track?
    """

    # --- Query OpenAI ---
    # Use the client's chat.completions.create method
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert AI optimizer and reinforcement learning consultant."},
            {"role": "user", "content": prompt.strip()}
        ]
    )

    print("\n🧠 GPT-4 Feedback:")
    # Access the response content using the new structure
    print(response.choices[0].message.content)

# --- Run loop ---
if __name__ == '__main__':
    while True:
        try:
            monitor_and_ask(log_dir)
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(poll_interval)