# --- ARP Leaderboard Tracker + Dashboard ---
# Automatically updates master leaderboard and visualizes top configs

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
LEADERBOARD_FILE = "results/master_leaderboard.csv"

# Load all CSVs in the results directory and summarize final performance
def build_leaderboard(results_dir=RESULTS_DIR):
    leaderboard = []
    for file in os.listdir(results_dir):
        if file.endswith(".csv") and not file.startswith("master"):
            path = os.path.join(results_dir, file)
            try:
                df = pd.read_csv(path)
                last = df.iloc[-1]
                summary = {
                    "Run": file.replace(".csv", ""),
                    "Final Train Accuracy": last["Train Accuracy"],
                    "Final Test Accuracy": last.get("Test Accuracy", 0.0),
                    "Min Train Loss": df["Train Loss"].min(),
                    "Switched to ARP": int(df["Switched to ARP"].any())
                }
                leaderboard.append(summary)
            except Exception as e:
                print(f"⚠️ Skipped {file}: {e}")
    return pd.DataFrame(leaderboard)

# Save leaderboard CSV
def save_leaderboard(df, leaderboard_file=LEADERBOARD_FILE):
    df_sorted = df.sort_values(by="Final Test Accuracy", ascending=False)
    df_sorted.to_csv(leaderboard_file, index=False)
    print(f"✅ Saved leaderboard to {leaderboard_file}")
    return df_sorted

# Visualize top runs
def plot_top_runs(df, metric="Final Test Accuracy", top_n=5):
    top_df = df.sort_values(by=metric, ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_df["Run"], top_df[metric], color="skyblue")
    plt.xlabel(metric)
    plt.title(f"Top {top_n} Runs by {metric}")
    plt.gca().invert_yaxis()
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va='center')
    plt.tight_layout()
    plt.show()

# Combined runner
def update_and_visualize_leaderboard():
    leaderboard_df = build_leaderboard()
    sorted_df = save_leaderboard(leaderboard_df)
    plot_top_runs(sorted_df, metric="Final Test Accuracy")
    plot_top_runs(sorted_df, metric="Min Train Loss")
    plot_top_runs(sorted_df, metric="Final Train Accuracy")

if __name__ == "__main__":
    update_and_visualize_leaderboard()
