"""
RealignR Leaderboard Dashboard Generator

This script builds on the leaderboard autoloader to create a comprehensive 
dashboard visualization of RealignR experiment results.

Features:
- Uses leaderboard_autoloader to compile experiment data
- Generates interactive visualizations and comparison charts
- Creates HTML dashboard with sortable tables and plots
- Enables filtering and side-by-side comparison of runs
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple

# Import our leaderboard autoloader
from leaderboard_autoloader import LeaderboardAutoloader


class LeaderboardDashboard:
    """Generates comprehensive dashboard visualizations from leaderboard data."""
    
    def __init__(self, 
                 leaderboard: Optional[LeaderboardAutoloader] = None,
                 base_dirs: List[str] = None,
                 primary_metric: str = "val_loss",
                 lower_is_better: bool = True):
        """
        Initialize the leaderboard dashboard generator.
        
        Args:
            leaderboard: Existing LeaderboardAutoloader instance (optional)
            base_dirs: List of base directories to scan if no leaderboard provided
            primary_metric: The metric to use for primary sorting/ranking
            lower_is_better: Whether lower values of the primary metric are better
        """
        if leaderboard is None:
            self.leaderboard = LeaderboardAutoloader(
                base_dirs=base_dirs or ["results", "runs"],
                primary_metric=primary_metric,
                lower_is_better=lower_is_better
            )
            self.leaderboard.compile_leaderboard()
        else:
            self.leaderboard = leaderboard
            
        self.df = self.leaderboard.df
        self.stats = self.leaderboard.stats
        self.output_dir = "dashboard"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up style for visualizations
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set(font_scale=1.2)
        self.colors = sns.color_palette("Set2", 8)
        
    def generate_comparison_plots(self, 
                                  metrics: List[str] = None,
                                  top_n: int = 5,
                                  by_dataset: bool = True) -> Dict[str, str]:
        """
        Generate comparison plots for top runs.
        
        Args:
            metrics: List of metrics to plot (if None, auto-detect)
            top_n: Number of top runs to include
            by_dataset: Whether to separate comparisons by dataset
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.df is None or len(self.df) == 0:
            print("No data available for plotting.")
            return {}
        
        # Auto-detect metrics if not specified
        if metrics is None:
            metrics = []
            # Look for common metrics
            for col in self.df.columns:
                if col.startswith(("best_", "final_")) and col not in [
                    "best_G_mean", "final_G_mean", "best_alpha", "final_alpha", "best_mu", "final_mu"
                ]:
                    metrics.append(col)
            # Limit to a reasonable number
            metrics = metrics[:6]
        
        plots = {}
        
        # Get grouping by dataset if requested
        if by_dataset and "dataset" in self.df.columns:
            datasets = self.df["dataset"].dropna().unique()
        else:
            datasets = [None]  # Single group with all runs
        
        for dataset in datasets:
            # Filter the dataframe for this dataset
            if dataset is not None:
                df_subset = self.df[self.df["dataset"] == dataset]
                dataset_name = dataset
            else:
                df_subset = self.df
                dataset_name = "all"
                
            if len(df_subset) == 0:
                continue
                
            # Get top N runs
            if "rank" in df_subset.columns:
                top_runs = df_subset.sort_values("rank").head(top_n)
            else:
                # If no rank, use primary metric
                primary_col = self._find_primary_metric_column()
                if primary_col:
                    top_runs = df_subset.sort_values(
                        primary_col, 
                        ascending=self.leaderboard.lower_is_better
                    ).head(top_n)
                else:
                    top_runs = df_subset.head(top_n)
            
            # Create bar charts for each metric
            for metric in metrics:
                if metric not in df_subset.columns:
                    continue
                    
                plt.figure(figsize=(12, 6))
                
                # Create a readable title
                metric_title = metric.replace("_", " ").title()
                if metric.startswith("best_"):
                    metric_title = "Best " + metric_title[5:]
                elif metric.startswith("final_"):
                    metric_title = "Final " + metric_title[6:]
                
                # Create bar chart
                ax = sns.barplot(
                    x="run_dir", 
                    y=metric, 
                    data=top_runs,
                    palette=self.colors
                )
                
                # Improve labels
                plt.title(f"{metric_title} - Top {len(top_runs)} Runs" + 
                          (f" ({dataset_name})" if dataset_name != "all" else ""), 
                          fontsize=14)
                plt.xlabel("")
                plt.ylabel(metric_title)
                
                # Rotate x labels for readability
                plt.xticks(rotation=45, ha="right")
                
                # Add value labels on bars
                for i, p in enumerate(ax.patches):
                    value = p.get_height()
                    if abs(value) < 0.01:
                        text = f"{value:.6f}"
                    else:
                        text = f"{value:.4f}"
                    ax.annotate(
                        text, 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', rotation=0, 
                        xytext=(0, 5), textcoords='offset points'
                    )
                
                plt.tight_layout()
                
                # Save the plot
                filename = f"{dataset_name}_{metric}_comparison.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=100, bbox_inches="tight")
                plt.close()
                
                plots[f"{dataset_name}_{metric}"] = filepath
                
            # If RealignR specific metrics exist, create special plots
            arp_metrics = ["final_G_mean", "final_alpha", "final_mu"]
            if all(m in df_subset.columns for m in arp_metrics):
                # Create comparison plot for ARP parameters
                plt.figure(figsize=(14, 7))
                
                # Prepare data
                arp_df = top_runs[arp_metrics + ["run_dir"]].melt(
                    id_vars=["run_dir"], 
                    value_vars=arp_metrics,
                    var_name="Parameter", 
                    value_name="Value"
                )
                
                # Clean parameter names
                arp_df["Parameter"] = arp_df["Parameter"].apply(
                    lambda x: x.replace("final_", "").replace("_", " ").title()
                )
                
                # Create grouped bar chart
                ax = sns.catplot(
                    x="run_dir", 
                    y="Value", 
                    hue="Parameter",
                    data=arp_df, 
                    kind="bar",
                    height=6, 
                    aspect=1.8,
                    palette=self.colors,
                    legend=False
                )
                
                # Add legend and labels
                plt.legend(title="Parameter", loc="upper right")
                plt.title(f"RealignR Parameters - Top Runs" + 
                         (f" ({dataset_name})" if dataset_name != "all" else ""), 
                         fontsize=14)
                plt.xlabel("")
                plt.ylabel("Parameter Value")
                
                # Rotate x labels for readability
                plt.xticks(rotation=45, ha="right")
                
                plt.tight_layout()
                
                # Save the plot
                filename = f"{dataset_name}_arp_params_comparison.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=100, bbox_inches="tight")
                plt.close()
                
                plots[f"{dataset_name}_arp_params"] = filepath
        
        return plots
    
    def generate_performance_matrix(self) -> Dict[str, str]:
        """
        Generate a matrix visualization comparing performance across datasets and models.
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.df is None or len(self.df) == 0:
            print("No data available for matrix plotting.")
            return {}
            
        plots = {}
        
        # Check if we have dataset and model columns
        if "dataset" not in self.df.columns or "model" not in self.df.columns:
            return plots
            
        # Find appropriate performance metric
        perf_metrics = [
            "best_val_loss", "final_val_loss", 
            "best_accuracy", "final_accuracy",
            "best_perplexity", "final_perplexity"
        ]
        
        perf_metric = None
        for metric in perf_metrics:
            if metric in self.df.columns:
                perf_metric = metric
                break
                
        if perf_metric is None:
            return plots
            
        # Create a pivot table
        try:
            # For each dataset-model pair, use the best run's performance
            pivot_df = self.df.loc[self.df.groupby(['dataset', 'model'])[perf_metric].idxmin()]
            matrix = pivot_df.pivot_table(
                index='dataset', 
                columns='model', 
                values=perf_metric,
                aggfunc='min'  # Take the best performance
            )
            
            # Create a heatmap
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(
                matrix, 
                annot=True, 
                fmt=".4f", 
                cmap="YlGnBu_r" if "loss" in perf_metric or "perplexity" in perf_metric else "YlGnBu",
                linewidths=.5
            )
            
            # Improve labels
            metric_title = perf_metric.replace("_", " ").title()
            if perf_metric.startswith("best_"):
                metric_title = "Best " + metric_title[5:]
            elif perf_metric.startswith("final_"):
                metric_title = "Final " + metric_title[6:]
                
            plt.title(f"{metric_title} by Dataset and Model", fontsize=14)
            plt.ylabel("Dataset")
            plt.xlabel("Model")
            
            plt.tight_layout()
            
            # Save the plot
            filename = f"performance_matrix_{perf_metric}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=100, bbox_inches="tight")
            plt.close()
            
            plots["performance_matrix"] = filepath
            
        except Exception as e:
            print(f"Error creating performance matrix: {e}")
            
        return plots
    
    def generate_realignr_advantage_plot(self) -> Dict[str, str]:
        """
        Generate plot showing RealignR's advantage over baselines.
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.df is None or len(self.df) == 0 or "is_realignr" not in self.df.columns:
            print("No data available for RealignR advantage plotting.")
            return {}
            
        plots = {}
        
        # Find appropriate performance metrics
        perf_metrics = [
            "best_val_loss", "final_val_loss", 
            "best_accuracy", "final_accuracy",
            "best_perplexity", "final_perplexity"
        ]
        
        available_metrics = [m for m in perf_metrics if m in self.df.columns]
        
        if not available_metrics:
            return plots
            
        for perf_metric in available_metrics:
            try:
                # Create a figure
                plt.figure(figsize=(12, 6))
                
                # Split by dataset if available
                if "dataset" in self.df.columns:
                    datasets = self.df["dataset"].dropna().unique()
                    
                    advantage_data = []
                    for dataset in datasets:
                        dataset_df = self.df[self.df["dataset"] == dataset]
                        
                        # Only include datasets with both RealignR and non-RealignR runs
                        realignr_runs = dataset_df[dataset_df["is_realignr"] == True]
                        baseline_runs = dataset_df[dataset_df["is_realignr"] != True]
                        
                        if len(realignr_runs) == 0 or len(baseline_runs) == 0:
                            continue
                            
                        # Get best performance for each
                        if "loss" in perf_metric or "perplexity" in perf_metric:
                            # Lower is better
                            realignr_best = realignr_runs[perf_metric].min()
                            baseline_best = baseline_runs[perf_metric].min()
                            # Calculate relative improvement (negative means RealignR is better)
                            rel_improvement = (realignr_best - baseline_best) / baseline_best
                            # Convert to percentage improvement
                            pct_improvement = -rel_improvement * 100
                        else:
                            # Higher is better (accuracy)
                            realignr_best = realignr_runs[perf_metric].max()
                            baseline_best = baseline_runs[perf_metric].max()
                            # Calculate relative improvement (positive means RealignR is better)
                            rel_improvement = (realignr_best - baseline_best) / baseline_best
                            # Convert to percentage improvement
                            pct_improvement = rel_improvement * 100
                            
                        advantage_data.append({
                            'dataset': dataset,
                            'improvement': pct_improvement,
                            'realignr_value': realignr_best,
                            'baseline_value': baseline_best
                        })
                        
                    if advantage_data:
                        advantage_df = pd.DataFrame(advantage_data)
                        
                        # Create bar chart
                        ax = sns.barplot(
                            x='dataset', 
                            y='improvement', 
                            data=advantage_df,
                            palette=['green' if x > 0 else 'red' for x in advantage_df['improvement']]
                        )
                        
                        # Add horizontal line at 0%
                        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                        
                        # Improve labels
                        metric_title = perf_metric.replace("_", " ").title()
                        if perf_metric.startswith("best_"):
                            metric_title = "Best " + metric_title[5:]
                        elif perf_metric.startswith("final_"):
                            metric_title = "Final " + metric_title[6:]
                            
                        plt.title(f"RealignR Advantage in {metric_title} by Dataset", fontsize=14)
                        plt.xlabel("Dataset")
                        plt.ylabel("% Improvement")
                        
                        # Add value labels on bars
                        for i, p in enumerate(ax.patches):
                            value = p.get_height()
                            dataset = advantage_df.iloc[i]
                            text = f"{value:.1f}%"
                            # Add values in annotation
                            realignr = dataset['realignr_value']
                            baseline = dataset['baseline_value']
                            detail = f"R:{realignr:.4f} vs B:{baseline:.4f}"
                            
                            ax.annotate(
                                text, 
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom' if p.get_height() > 0 else 'top', 
                                xytext=(0, 8 if p.get_height() > 0 else -8), 
                                textcoords='offset points'
                            )
                            
                            ax.annotate(
                                detail, 
                                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                                ha='center', va='center', 
                                rotation=90, 
                                fontsize=8
                            )
                        
                        plt.tight_layout()
                        
                        # Save the plot
                        filename = f"realignr_advantage_{perf_metric}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        plt.savefig(filepath, dpi=100, bbox_inches="tight")
                        plt.close()
                        
                        plots[f"realignr_advantage_{perf_metric}"] = filepath
                
            except Exception as e:
                print(f"Error creating RealignR advantage plot for {perf_metric}: {e}")
            
        return plots
    
    def generate_html_dashboard(self, 
                               plots: Dict[str, str] = None, 
                               output_file: str = None) -> str:
        """
        Generate an HTML dashboard with tables and plots.
        
        Args:
            plots: Dictionary of plots generated by other methods
            output_file: Path to save the HTML file
            
        Returns:
            Path to the saved HTML file
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.df is None or len(self.df) == 0:
            print("No data available for dashboard.")
            return None
            
        if plots is None:
            plots = {}
            # Generate standard plots
            plots.update(self.generate_comparison_plots())
            plots.update(self.generate_performance_matrix())
            plots.update(self.generate_realignr_advantage_plot())
            
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"dashboard_{self.timestamp}.html")
        
        # Start building HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RealignR Leaderboard Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .header {{
                    background-color: #3b6ea5;
                    color: white;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    border-radius: 5px;
                }}
                .section {{
                    background-color: white;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 1rem;
                }}
                th, td {{
                    padding: 8px 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                    cursor: pointer;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .plot-container {{
                    margin: 1rem 0;
                    text-align: center;
                }}
                .plot-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .stats {{
                    margin-bottom: 1rem;
                }}
                .stats-item {{
                    margin-bottom: 0.5rem;
                }}
                .generated-at {{
                    font-size: 0.8rem;
                    color: #666;
                    text-align: center;
                    margin-top: 2rem;
                }}
                /* Add styles for filtering and sorting */
                .filters {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-bottom: 1rem;
                }}
                .filter-group {{
                    display: flex;
                    flex-direction: column;
                }}
                .filter-label {{
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .filter-select {{
                    padding: 5px;
                    border-radius: 3px;
                    border: 1px solid #ddd;
                }}
                /* Two-column layout for plots */
                .plot-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                    gap: 20px;
                }}
            </style>
            <!-- Add sortable tables JavaScript -->
            <script>
                function sortTable(n, tableId) {{
                    var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
                    table = document.getElementById(tableId);
                    switching = true;
                    dir = "asc";
                    
                    while (switching) {{
                        switching = false;
                        rows = table.rows;
                        
                        for (i = 1; i < (rows.length - 1); i++) {{
                            shouldSwitch = false;
                            x = rows[i].getElementsByTagName("TD")[n];
                            y = rows[i + 1].getElementsByTagName("TD")[n];
                            
                            // Check if the two rows should switch
                            if (dir == "asc") {{
                                if (isNaN(x.innerHTML) ? 
                                    x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase() :
                                    parseFloat(x.innerHTML) > parseFloat(y.innerHTML)) {{
                                    shouldSwitch = true;
                                    break;
                                }}
                            }} else if (dir == "desc") {{
                                if (isNaN(x.innerHTML) ? 
                                    x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase() :
                                    parseFloat(x.innerHTML) < parseFloat(y.innerHTML)) {{
                                    shouldSwitch = true;
                                    break;
                                }}
                            }}
                        }}
                        
                        if (shouldSwitch) {{
                            rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                            switching = true;
                            switchcount++;
                        }} else {{
                            if (switchcount == 0 && dir == "asc") {{
                                dir = "desc";
                                switching = true;
                            }}
                        }}
                    }}
                }}
                
                function filterTable() {{
                    var dataset = document.getElementById("filter-dataset").value;
                    var model = document.getElementById("filter-model").value;
                    var realignr = document.getElementById("filter-realignr").value;
                    
                    var table = document.getElementById("leaderboard-table");
                    var tr = table.getElementsByTagName("tr");
                    
                    for (var i = 1; i < tr.length; i++) {{
                        var show = true;
                        var row = tr[i];
                        
                        // Check dataset filter
                        if (dataset && dataset !== "all") {{
                            var datasetCell = row.querySelector("td[data-dataset]");
                            if (!datasetCell || datasetCell.getAttribute("data-dataset") !== dataset) {{
                                show = false;
                            }}
                        }}
                        
                        // Check model filter
                        if (show && model && model !== "all") {{
                            var modelCell = row.querySelector("td[data-model]");
                            if (!modelCell || modelCell.getAttribute("data-model") !== model) {{
                                show = false;
                            }}
                        }}
                        
                        // Check realignr filter
                        if (show && realignr && realignr !== "all") {{
                            var realignrVal = row.querySelector("td[data-realignr]").getAttribute("data-realignr");
                            if ((realignr === "yes" && realignrVal !== "true") || 
                                (realignr === "no" && realignrVal !== "false")) {{
                                show = false;
                            }}
                        }}
                        
                        row.style.display = show ? "" : "none";
                    }}
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>RealignR Leaderboard Dashboard</h1>
                </div>
                
                <div class="section">
                    <h2>Summary Statistics</h2>
                    <div class="stats">
        """
        
        # Add summary statistics
        if self.stats:
            html += f"<div class='stats-item'><strong>Total Runs:</strong> {self.stats.get('total_runs', 0)}</div>"
            
            if "dataset_counts" in self.stats:
                html += "<div class='stats-item'><strong>Datasets:</strong> "
                for dataset, count in self.stats["dataset_counts"].items():
                    html += f"{dataset} ({count} runs), "
                html = html.rstrip(", ") + "</div>"
                
            if "model_counts" in self.stats:
                html += "<div class='stats-item'><strong>Models:</strong> "
                for model, count in self.stats["model_counts"].items():
                    html += f"{model} ({count} runs), "
                html = html.rstrip(", ") + "</div>"
                
            if "realignr_count" in self.stats:
                html += f"<div class='stats-item'><strong>RealignR Runs:</strong> {self.stats['realignr_count']}</div>"
                html += f"<div class='stats-item'><strong>Non-RealignR Runs:</strong> {self.stats['non_realignr_count']}</div>"
                
            if "best_run" in self.stats:
                html += "<div class='stats-item'><strong>Best Run:</strong> "
                best_run = self.stats["best_run"]
                for key, value in best_run.items():
                    if key not in ["rank", "delta_from_best"] and not key.startswith("config_"):
                        html += f"{key}: {value}, "
                html = html.rstrip(", ") + "</div>"
        
        html += """
                    </div>
                </div>
                
                <div class="section">
                    <h2>Leaderboard Table</h2>
        """
        
        # Add filters if we have filterable columns
        filter_columns = []
        if "dataset" in self.df.columns:
            filter_columns.append(("dataset", "Dataset", self.df["dataset"].dropna().unique()))
        if "model" in self.df.columns:
            filter_columns.append(("model", "Model", self.df["model"].dropna().unique()))
        if "is_realignr" in self.df.columns:
            filter_columns.append(("realignr", "RealignR", [("yes", "Yes"), ("no", "No")]))
            
        if filter_columns:
            html += """
                    <div class="filters">
            """
            
            for col_id, col_name, options in filter_columns:
                html += f"""
                        <div class="filter-group">
                            <div class="filter-label">{col_name}:</div>
                            <select id="filter-{col_id}" class="filter-select" onchange="filterTable()">
                                <option value="all">All</option>
                """
                
                # Add options
                if isinstance(options, list) and isinstance(options[0], tuple):
                    # Pre-formatted options with values and labels
                    for value, label in options:
                        html += f'<option value="{value}">{label}</option>'
                else:
                    # Regular options where value = label
                    for option in sorted(options):
                        html += f'<option value="{option}">{option}</option>'
                
                html += """
                            </select>
                        </div>
                """
                
            html += """
                    </div>
            """
        
        # Create leaderboard table
        html += """
                    <div class="table-responsive">
                        <table id="leaderboard-table">
                            <thead>
                                <tr>
        """
        
        # Select columns to display
        display_columns = []
        if "rank" in self.df.columns:
            display_columns.append(("rank", "Rank"))
            
        # Add other common columns
        common_columns = [
            ("dataset", "Dataset"),
            ("model", "Model"),
            ("is_realignr", "RealignR"),
        ]
        
        for col_id, col_name in common_columns:
            if col_id in self.df.columns:
                display_columns.append((col_id, col_name))
                
        # Add performance metrics
        perf_metrics = []
        for col in self.df.columns:
            if col.startswith(("best_", "final_")) and col not in [
                "best_G_mean", "final_G_mean", "best_alpha", "final_alpha", 
                "best_mu", "final_mu", "best_CPR_trigger", "final_CPR_trigger"
            ]:
                # Create a readable name
                if col.startswith("best_"):
                    col_name = "Best " + col[5:].replace("_", " ").title()
                else:
                    col_name = "Final " + col[6:].replace("_", " ").title()
                    
                perf_metrics.append((col, col_name))
                
        # Sort metrics by name
        perf_metrics.sort(key=lambda x: x[1])
        display_columns.extend(perf_metrics)
        
        # Add ARP metrics
        arp_metrics = [
            ("final_G_mean", "G Mean"),
            ("final_alpha", "Alpha"),
            ("final_mu", "Mu"),
        ]
        
        for col_id, col_name in arp_metrics:
            if col_id in self.df.columns:
                display_columns.append((col_id, col_name))
                
        # Add run info
        run_info_columns = [
            ("total_steps", "Steps"),
            ("total_epochs", "Epochs"),
            ("run_date", "Date"),
            ("run_dir", "Run Directory"),
        ]
        
        for col_id, col_name in run_info_columns:
            if col_id in self.df.columns:
                display_columns.append((col_id, col_name))
        
        # Add table headers
        for i, (col_id, col_name) in enumerate(display_columns):
            html += f'<th onclick="sortTable({i}, \'leaderboard-table\')">{col_name}</th>'
            
        html += """
                                </tr>
                            </thead>
                            <tbody>
        """
        
        # Add table rows
        for _, row in self.df.iterrows():
            html += "<tr>"
            
            for col_id, _ in display_columns:
                value = row.get(col_id, "")
                
                # Format the value
                if isinstance(value, float):
                    if abs(value) < 0.01:
                        formatted = f"{value:.6f}"
                    else:
                        formatted = f"{value:.4f}"
                elif pd.isna(value):
                    formatted = ""
                else:
                    formatted = str(value)
                    
                # Add data attributes for filtering
                data_attrs = ""
                if col_id == "dataset" and value:
                    data_attrs = f' data-dataset="{value}"'
                elif col_id == "model" and value:
                    data_attrs = f' data-model="{value}"'
                elif col_id == "is_realignr":
                    data_attrs = f' data-realignr="{str(value).lower()}"'
                    formatted = "Yes" if value else "No"
                    
                html += f'<td{data_attrs}>{formatted}</td>'
                
            html += "</tr>"
            
        html += """
                            </tbody>
                        </table>
                    </div>
                </div>
        """
        
        # Add plot sections
        if plots:
            html += """
                <div class="section">
                    <h2>Performance Visualizations</h2>
                    <div class="plot-grid">
            """
            
            # Add performance comparison plots
            for plot_id, plot_path in plots.items():
                if "advantage" in plot_id or "matrix" in plot_id:
                    continue  # These get their own sections
                
                # Get a readable title
                if plot_id.endswith(("_comparison.png", "_params_comparison.png")):
                    plot_name = plot_id.replace("_", " ").title().replace(".Png", "")
                else:
                    plot_parts = plot_id.split('_')
                    metric_part = '_'.join(plot_parts[1:]) if len(plot_parts) > 1 else plot_parts[0]
                    dataset_part = plot_parts[0] if len(plot_parts) > 1 else "All"
                    
                    plot_name = f"{metric_part.replace('_', ' ').title()} ({dataset_part})"
                
                # Calculate a relative path to the image
                rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))
                
                html += f"""
                        <div class="plot-container">
                            <h3>{plot_name}</h3>
                            <img src="{rel_path}" alt="{plot_name}">
                        </div>
                """
            
            html += """
                    </div>
                </div>
            """
            
            # Add RealignR advantage section if those plots exist
            advantage_plots = {k: v for k, v in plots.items() if "advantage" in k}
            if advantage_plots:
                html += """
                    <div class="section">
                        <h2>RealignR Advantage Analysis</h2>
                        <div class="plot-grid">
                """
                
                for plot_id, plot_path in advantage_plots.items():
                    metric_name = plot_id.replace("realignr_advantage_", "").replace("_", " ").title()
                    rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))
                    
                    html += f"""
                            <div class="plot-container">
                                <h3>RealignR Advantage: {metric_name}</h3>
                                <img src="{rel_path}" alt="RealignR Advantage: {metric_name}">
                            </div>
                    """
                
                html += """
                        </div>
                    </div>
                """
                
            # Add performance matrix section if it exists
            matrix_plots = {k: v for k, v in plots.items() if "matrix" in k}
            if matrix_plots:
                html += """
                    <div class="section">
                        <h2>Performance Matrix Analysis</h2>
                        <div class="plot-grid">
                """
                
                for plot_id, plot_path in matrix_plots.items():
                    metric_name = plot_id.replace("performance_matrix_", "").replace("_", " ").title()
                    if not metric_name:
                        metric_name = "Overall Performance"
                        
                    rel_path = os.path.relpath(plot_path, os.path.dirname(output_file))
                    
                    html += f"""
                            <div class="plot-container">
                                <h3>Performance Matrix: {metric_name}</h3>
                                <img src="{rel_path}" alt="Performance Matrix: {metric_name}">
                            </div>
                    """
                
                html += """
                        </div>
                    </div>
                """
        
        # Add footer
        html += f"""
                <div class="generated-at">
                    Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_file, 'w') as f:
            f.write(html)
            
        print(f"Dashboard generated at {output_file}")
        return output_file
    
    def _find_primary_metric_column(self) -> Optional[str]:
        """Find the primary metric column in the DataFrame."""
        primary_metric = self.leaderboard.primary_metric
        
        if primary_metric in self.df.columns:
            return primary_metric
        elif f"best_{primary_metric}" in self.df.columns:
            return f"best_{primary_metric}"
        elif f"final_{primary_metric}" in self.df.columns:
            return f"final_{primary_metric}"
        
        # Try to find a reasonable default
        for prefix in ["best_", "final_"]:
            for metric in ["val_loss", "test_loss", "accuracy", "perplexity"]:
                col = f"{prefix}{metric}"
                if col in self.df.columns:
                    return col
                    
        return None


def main():
    """CLI entrypoint for the dashboard generator."""
    parser = argparse.ArgumentParser(description="RealignR Leaderboard Dashboard Generator")
    parser.add_argument("--dirs", nargs="+", default=["results", "runs"],
                       help="Base directories to scan")
    parser.add_argument("--metric", default="val_loss",
                       help="Primary metric for ranking")
    parser.add_argument("--higher-better", action="store_true",
                       help="Set if higher values of metric are better")
    parser.add_argument("--output-dir", default="dashboard",
                       help="Directory to save dashboard files")
    parser.add_argument("--html", default=None,
                       help="Custom name for HTML dashboard file")
    
    args = parser.parse_args()
    
    # Create leaderboard
    leaderboard = LeaderboardAutoloader(
        base_dirs=args.dirs,
        primary_metric=args.metric,
        lower_is_better=not args.higher_better
    )
    
    # Compile data
    leaderboard.compile_leaderboard()
    
    # Create dashboard
    dashboard = LeaderboardDashboard(leaderboard)
    dashboard.output_dir = args.output_dir
    
    # Generate plots
    plots = {}
    plots.update(dashboard.generate_comparison_plots())
    plots.update(dashboard.generate_performance_matrix())
    plots.update(dashboard.generate_realignr_advantage_plot())
    
    # Generate HTML
    if args.html:
        html_path = os.path.join(args.output_dir, args.html)
    else:
        html_path = None
        
    dashboard.generate_html_dashboard(plots, html_path)
    
    # Print summary
    leaderboard.print_summary()
    
    return dashboard


if __name__ == "__main__":
    main()
