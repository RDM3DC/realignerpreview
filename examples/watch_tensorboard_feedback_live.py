RUN_DIR = "runs/cpr_reset_sanity"
# watch_tensorboard_feedback_live.py (with TensorBoard scalar feedback logging)

import os
import time
import openai
import json
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get the latest run directory using glob pattern
def get_latest_run_dir():
    # Use the fixed run directory if specified
    if os.path.exists(RUN_DIR):
        print(f"🔍 Using fixed run directory: {RUN_DIR}")
        return RUN_DIR
    else:
        print(f"⚠️ Specified run directory {RUN_DIR} does not exist.")
        return None

poll_interval = 60
control_file = "realignr_control.json"
watch_tags = ["Accuracy/train", "Loss/train", "G_mean", "alpha", "mu", "CPR_trigger"]

# --- Live scalar summary + control update ---
def analyze_feedback_log(log_dir):
    feedback_path = os.path.join(log_dir, "feedback_log.json")
    alpha_vals, mu_vals, cpr_events = [], [], 0
    try:
        with open(feedback_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                if 'alpha' in record and 'mu' in record:
                    alpha_vals.append(record['alpha'])
                    mu_vals.append(record['mu'])
                if record.get('apply_cpr', False):
                    cpr_events += 1
    except FileNotFoundError:
        return ""

    def summarize(values, name):
        if not values: return f"- No history for {name}\n"
        return f"- {name} avg: {sum(values)/len(values):.4f}, range: {min(values):.4f}-{max(values):.4f}\n"

    return (
        "Past GPT feedback summary:\n"
        + summarize(alpha_vals, "alpha")
        + summarize(mu_vals, "mu")
        + f"- CPR triggered: {cpr_events} times in prior epochs\n"
    )

def load_phase_schedule():
    try:
        with open(os.path.join(os.path.dirname(__file__), '..', 'phase_schedule.json'), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_current_phase(epoch, schedule):
    current = {"name": "default", "params": {}}
    for entry in schedule:
        if epoch >= entry.get("start_epoch", 0):
            current = entry
        else:
            break
    return current

def monitor_and_ask(log_dir):
    print("🔍 Monitoring TensorBoard logs with GPT feedback + control output...")
    ea = EventAccumulator(log_dir)
    ea.Reload()

    metrics = {}
    epoch_guess = None
    for tag in watch_tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            if events:
                metrics[tag] = events[-1].value
            if tag == "Accuracy/train":
                epoch_guess = events[-1].step

    if not metrics:
        print("⚠️ No scalars available yet. Waiting...")
        return

    report = "".join([f"{k}: {v:.4f}\n" for k, v in metrics.items()])
    feedback_summary = analyze_feedback_log(log_dir)
    phase_schedule = load_phase_schedule()
    current_phase = get_current_phase(epoch_guess or 0, phase_schedule)
    phase_name = current_phase.get("name", "default")
    phase_params = current_phase.get("params", {})

    phase_summary = "\n".join([f"- {k}: {v}" for k, v in phase_params.items()]) if phase_params else "- No special parameters."

    feedback_summary = analyze_feedback_log(log_dir)
    prompt = f"""
Current Phase: {phase.upper()} - {reason}

SYSTEM: You are an AI flight engineer guiding a memory-based optimizer called RealignR. It switches from AdamW to ARP at epoch 100. ARP has a resistance memory buffer G_ij and is controlled by parameters alpha and mu. CPR fires if accuracy < 2% and G_mean < 0.01, triggering a soft reset (G *= 0.2).

{feedback_summary}

Helpful guidance:
- If CPR fires multiple times in a row, suggest lowering alpha and increasing mu temporarily.
- If G_mean is low but accuracy is holding, do not trigger CPR.
- If accuracy is dropping but G_mean is high, lower mu slightly and do not CPR yet.
- CPR should only trigger once, then allow optimizer to stabilize.
- After CPR, ramp alpha back up gradually over 5 epochs.

🔍 Observations:
    Steady Improvement:

    Accuracy consistently climbs, from 87.34% (Epoch 2) to around 90.27% (Epoch 23).

    Loss decreases from 0.4071 (Epoch 2) down to approximately 0.3231 (Epoch 23).

    Plateaus & CPR (Dynamic Adjustments):

    Loss occasionally plateaus, triggering "CPR"—adaptive adjustments to ARP's parameters (alpha and mu).

    Commonly chosen CPR adjustments:

    Alpha: alternating around 0.0078 - 0.011

    Mu: predominantly stable at 0.001, with occasional adjustments to 0.0012.

    Gradient Norm Spikes (GradNorm: inf):

    Notable inf spikes at Epochs 10 and 12, suggesting transient instability or exploding gradients.

    Accuracy Variance Reduction:

    Variance (Var) consistently decreases from ~0.110 to around 0.088, signifying improved training stability and smoother gradient updates.

    🚩 Key Areas of Concern:
    Gradient instability (GradNorm: inf):
    Indicates potential numerical issues or overly aggressive parameter updates.

    Frequent CPR triggers:
    Constant parameter tweaking might signal too-sensitive conditions for plateau detection.

    🛠️ Recommended Actions:
    Adjust CPR Sensitivity:

    Loosen CPR conditions slightly (e.g., fewer triggers or longer intervals between adjustments).

    Allow the network more epochs between CPR interventions to naturally stabilize.

    Gradient Norm Management:

    Consider gradient clipping or norm regularization to prevent inf values.

    Slightly reduce initial alpha or scale it down dynamically when large gradient norms occur.

    Parameter Adjustment Strategy:

    Narrow down dynamic range for alpha (e.g., 0.008 - 0.010) to reduce volatility.

    Keep mu fixed at 0.001 for stability, adjusting only when necessary.

OUTPUT JSON ONLY in this format:
{{ "alpha": 0.0025, "mu": 0.001, "apply_cpr": false }}

Phase: {phase_name}
Phase Parameters:
{phase_summary}

Scalars:
    {report}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Respond with optimizer control settings only."},
            {"role": "user", "content": prompt.strip()}
        ]
    )

    reply = response.choices[0].message.content.strip()
    print("\n🧠 GPT Response:")
    print(reply)

    try:
        control = json.loads(reply)
        control_path = os.path.join(os.path.dirname(__file__), '..', control_file)
        with open(control_path, "w") as f:
            json.dump(control, f)
        print(f"✅ Control written to {control_path}")

        # Also log alpha/mu and CPR_trigger as TensorBoard feedback
        tb_feedback_path = os.path.join(log_dir, "feedback_log.json")
        with open(tb_feedback_path, "a") as f:
            control["epoch"] = time.time()
            json.dump(control, f)
            f.write("\n")

    except Exception as e:
        print(f"⚠️ Failed to parse GPT response: {e}")

if __name__ == '__main__':
    # === Load phase from meta-controller ===
    phase = "stable"
    reason = ""
    if os.path.exists("realignr_phase.json"):
        try:
            with open("realignr_phase.json", "r") as f:
                phase_data = json.load(f)
                phase = phase_data.get("phase", "stable")
                reason = phase_data.get("reason", "")
        except Exception as e:
            print(f"⚠️ Failed to read phase: {e}")

    while True:
        latest_log_dir = get_latest_run_dir()
        if latest_log_dir:
            try:
                monitor_and_ask(latest_log_dir)
            except Exception as e:
                print(f"Error during monitoring: {e}")
        else:
            print("⏳ No matching run directories found. Waiting...")

        time.sleep(poll_interval)
