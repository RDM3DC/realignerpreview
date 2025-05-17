"""
RealignR Dataset Switching Leaderboard

This script demonstrates how to use the leaderboard system to analyze
training runs with dataset switching, a key capability mentioned in the requirements.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from leaderboard_autoloader import LeaderboardAutoloader
from leaderboard_dashboard import LeaderboardDashboard

def analyze_dataset_switching():
    """
    Analyze training runs that involve dataset switching.
    
    This function demonstrates how to:
    1. Find experiments with dataset switching
    2. Extract metrics before and after switching
    3. Compare performance across different switching strategies
    4. Generate visualizations specific to dataset switching
    """
    print("=== RealignR Dataset Switching Analysis ===")
    
    # Create a specialized leaderboard autoloader with patterns that match dataset switching
    loader = LeaderboardAutoloader(
        base_dirs=["runs", "results"],
        primary_metric="val_loss",
        lower_is_better=True,
        include_patterns=["*dataset_switch*", "*multi_dataset*"]
    )
    
    # Compile the leaderboard
    df = loader.compile_leaderboard()
    
    if len(df) == 0:
        print("No dataset switching experiments found.")
        return
    
    print(f"Found {len(df)} experiments with dataset switching.")
    
    # Extract and process dataset switch information
    extract_dataset_switch_info(df)
    
    # Save processed data
    dataset_switch_csv = "dataset_switch_leaderboard.csv"
    df.to_csv(dataset_switch_csv, index=False)
    print(f"Saved dataset switching leaderboard to {dataset_switch_csv}")
    
    # Generate specialized visualizations
    generate_switch_visualizations(df)
    
def extract_dataset_switch_info(df):
    """
    Extract detailed information about dataset switching from experiment files.
    
    Args:
        df: DataFrame from the leaderboard
    """
    # Add columns for dataset switching information
    df['primary_dataset'] = None
    df['secondary_dataset'] = None
    df['switch_step'] = None
    df['pre_switch_loss'] = None
    df['post_switch_loss'] = None
    df['recovery_rate'] = None
    
    # Process each experiment
    for idx, row in df.iterrows():
        # Extract dataset switching info from run name or config
        if 'run_name' in row:
            # Try to extract from run name using regex patterns
            primary_match = re.search(r'from[-_]?(\w+)', row['run_name'], re.IGNORECASE)
            secondary_match = re.search(r'to[-_]?(\w+)', row['run_name'], re.IGNORECASE)
            switch_match = re.search(r'switch[-_]?(?:at|step)?[-_]?(\d+)', row['run_name'], re.IGNORECASE)
            
            if primary_match:
                df.at[idx, 'primary_dataset'] = primary_match.group(1)
            if secondary_match:
                df.at[idx, 'secondary_dataset'] = secondary_match.group(1)
            if switch_match:
                df.at[idx, 'switch_step'] = int(switch_match.group(1))
        
        # Look at config parameters if available
        for col in df.columns:
            if col.startswith('config_'):
                param_name = col[7:]  # Remove 'config_' prefix
                
                # Check for dataset switch parameters
                if 'primary_dataset' in param_name.lower() or 'first_dataset' in param_name.lower():
                    df.at[idx, 'primary_dataset'] = row[col]
                elif 'secondary_dataset' in param_name.lower() or 'second_dataset' in param_name.lower():
                    df.at[idx, 'secondary_dataset'] = row[col]
                elif 'switch_step' in param_name.lower() or 'switch_at' in param_name.lower():
                    df.at[idx, 'switch_step'] = row[col]
        
        # Try to extract performance before and after switch
        if 'full_path' in row and row['full_path'] and os.path.exists(row['full_path']):
            try:
                metrics_df = pd.read_csv(row['full_path'])
                
                # Find step column
                step_col = None
                for col in metrics_df.columns:
                    if col.lower() in ['step', 'iteration', 'steps']:
                        step_col = col
                        break
                
                if step_col and 'switch_step' in df.columns and df.at[idx, 'switch_step'] is not None:
                    switch_step = df.at[idx, 'switch_step']
                    
                    # Find loss or other performance metrics
                    for metric in ['val_loss', 'loss', 'train_loss', 'test_loss']:
                        if metric in metrics_df.columns:
                            # Get performance before switch
                            pre_switch = metrics_df[metrics_df[step_col] < switch_step]
                            if not pre_switch.empty:
                                df.at[idx, 'pre_switch_loss'] = pre_switch[metric].iloc[-1]
                            
                            # Get performance after switch
                            post_switch = metrics_df[metrics_df[step_col] >= switch_step]
                            if not post_switch.empty:
                                df.at[idx, 'post_switch_loss'] = post_switch[metric].iloc[-1]
                            
                            # Calculate recovery rate (how quickly it adapts to new dataset)
                            if not pre_switch.empty and not post_switch.empty and len(post_switch) > 1:
                                pre_val = pre_switch[metric].iloc[-1]
                                post_initial = post_switch[metric].iloc[0]
                                post_final = post_switch[metric].iloc[-1]
                                
                                if post_initial > pre_val:  # Performance initially got worse (expected)
                                    recovery = (post_initial - post_final) / (post_initial - pre_val)
                                    df.at[idx, 'recovery_rate'] = max(0, min(1, recovery))
                            
                            break  # Found a metric
            except Exception as e:
                print(f"Error processing metrics file {row['full_path']}: {e}")

def generate_switch_visualizations(df):
    """
    Generate visualizations specific to dataset switching.
    
    Args:
        df: DataFrame with dataset switching information
    """
    # Create output directory
    os.makedirs("dataset_switch_analysis", exist_ok=True)
    
    # 1. Compare recovery rate by optimizer
    if 'recovery_rate' in df.columns and 'optimizer_normalized' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Filter to runs with recovery rate
        df_recovery = df[df['recovery_rate'].notna()]
        
        if not df_recovery.empty:
            ax = sns.barplot(x='optimizer_normalized', y='recovery_rate', data=df_recovery)
            
            plt.title('Dataset Switch Recovery Rate by Optimizer', fontsize=14)
            plt.xlabel('Optimizer')
            plt.ylabel('Recovery Rate (higher is better)')
            plt.ylim(0, 1.05)
            
            # Add value labels on bars
            for i, p in enumerate(ax.patches):
                ax.annotate(
                    f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', 
                    xytext=(0, 5), textcoords='offset points'
                )
            
            plt.tight_layout()
            plt.savefig("dataset_switch_analysis/recovery_by_optimizer.png", dpi=100, bbox_inches="tight")
            plt.close()
    
    # 2. Performance change after switch
    if 'pre_switch_loss' in df.columns and 'post_switch_loss' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Filter to runs with both metrics
        df_change = df[(df['pre_switch_loss'].notna()) & (df['post_switch_loss'].notna())]
        
        if not df_change.empty:
            # Calculate relative change
            df_change['relative_change'] = (df_change['post_switch_loss'] - df_change['pre_switch_loss']) / df_change['pre_switch_loss']
            
            # Sort by performance impact (smaller impact is better)
            df_change = df_change.sort_values('relative_change')
            
            # Create labels combining run name and optimizer
            if 'run_name' in df_change.columns and 'optimizer_normalized' in df_change.columns:
                df_change['run_label'] = df_change.apply(
                    lambda x: f"{x['run_name']} ({x['optimizer_normalized']})", axis=1
                )
            else:
                df_change['run_label'] = df_change.index
            
            # Create the plot
            colors = ['green' if x <= 0 else 'red' for x in df_change['relative_change']]
            ax = sns.barplot(x='relative_change', y='run_label', data=df_change, palette=colors)
            
            plt.title('Performance Impact After Dataset Switch (Relative Change in Loss)', fontsize=14)
            plt.xlabel('Relative Change (negative is better)')
            plt.ylabel('')
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels on bars
            for i, p in enumerate(ax.patches):
                ax.annotate(
                    f"{p.get_width():.2%}", 
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left' if p.get_width() >= 0 else 'right', va='center', 
                    xytext=(5 if p.get_width() >= 0 else -5, 0), 
                    textcoords='offset points'
                )
            
            plt.tight_layout()
            plt.savefig("dataset_switch_analysis/performance_impact.png", dpi=100, bbox_inches="tight")
            plt.close()
    
    # 3. Compare primary to secondary dataset performance
    if 'primary_dataset' in df.columns and 'secondary_dataset' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Group by dataset pairs
        dataset_pairs = df.groupby(['primary_dataset', 'secondary_dataset'])
        
        data = []
        for (primary, secondary), group in dataset_pairs:
            if primary and secondary:
                # For each optimizer, find the best run
                if 'optimizer_normalized' in group.columns:
                    for optimizer, opt_group in group.groupby('optimizer_normalized'):
                        if 'pre_switch_loss' in opt_group.columns and 'post_switch_loss' in opt_group.columns:
                            best_run = opt_group.sort_values('post_switch_loss').iloc[0]
                            
                            if pd.notna(best_run.get('pre_switch_loss')) and pd.notna(best_run.get('post_switch_loss')):
                                data.append({
                                    'dataset_pair': f"{primary}→{secondary}",
                                    'optimizer': optimizer,
                                    'pre_switch_loss': best_run['pre_switch_loss'],
                                    'post_switch_loss': best_run['post_switch_loss']
                                })
        
        if data:
            pair_df = pd.DataFrame(data)
            
            # Plot grouped bar chart
            plt.figure(figsize=(14, 8))
            ax = sns.catplot(
                x='dataset_pair', 
                y='post_switch_loss', 
                hue='optimizer',
                data=pair_df, 
                kind='bar',
                height=6, 
                aspect=1.5
            )
            
            plt.title('Post-Switch Performance by Dataset Pair', fontsize=14)
            plt.xlabel('Dataset Transition')
            plt.ylabel('Post-Switch Loss (lower is better)')
            
            plt.tight_layout()
            plt.savefig("dataset_switch_analysis/dataset_pair_performance.png", dpi=100, bbox_inches="tight")
            plt.close()
    
    print("Generated visualizations in the 'dataset_switch_analysis' directory.")

if __name__ == "__main__":
    analyze_dataset_switching()
