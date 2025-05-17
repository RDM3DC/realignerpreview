"""
RealignR Leaderboard Autoloader

This module scans result directories to automatically compile a leaderboard
of RealignR training runs, extracting key metrics and metadata.

Features:
- Recursive scanning of results directories
- Parsing metrics from CSV files
- Extracting metadata from filenames and run configs
- Aggregation into a sortable DataFrame 
- Ranking and comparison across runs
- Export to CSV, JSON, or Markdown table
"""

import os
import glob
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime


class LeaderboardAutoloader:
    """Main class for loading and processing RealignR experiment results."""
    
    def __init__(self, 
                 base_dirs: List[str] = None,
                 primary_metric: str = "val_loss",
                 lower_is_better: bool = True,
                 include_patterns: List[str] = None,
                 exclude_patterns: List[str] = None):
        """
        Initialize the leaderboard autoloader.
        
        Args:
            base_dirs: List of base directories to scan (defaults to ["results", "runs"])
            primary_metric: The metric to use for primary sorting/ranking
            lower_is_better: Whether lower values of the primary metric are better
            include_patterns: Glob patterns to include (e.g., ["**/metrics.csv"])
            exclude_patterns: Glob patterns to exclude (e.g., ["*_temp_*"])
        """
        self.base_dirs = base_dirs or ["results", "runs"]
        self.primary_metric = primary_metric
        self.lower_is_better = lower_is_better
        self.include_patterns = include_patterns or ["**/metrics.csv", "**/stats.csv", "**/final_metrics.csv"]
        self.exclude_patterns = exclude_patterns or ["**/temp/*", "**/tmp/*"]
        
        self.df = None  # Will hold the compiled DataFrame
        self.stats = {}  # Will hold summary statistics
        
    def scan_directories(self) -> List[str]:
        """
        Scan all base directories for CSV files matching patterns.
        
        Returns:
            List of file paths to relevant CSV files
        """
        all_files = []
        
        for base_dir in self.base_dirs:
            if not os.path.exists(base_dir):
                print(f"Warning: Directory {base_dir} does not exist, skipping.")
                continue
                
            # Apply include patterns
            for pattern in self.include_patterns:
                pattern_path = os.path.join(base_dir, pattern)
                matched_files = glob.glob(pattern_path, recursive=True)
                all_files.extend(matched_files)
        
        # Apply exclude patterns
        for pattern in self.exclude_patterns:
            all_files = [f for f in all_files if not glob.fnmatch.fnmatch(f, pattern)]
            
        print(f"Found {len(all_files)} metric files across {len(self.base_dirs)} base directories.")
        return all_files
    
    def extract_metadata_from_path(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from file path components.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary of extracted metadata
        """
        path_obj = Path(file_path)
        
        # Initialize metadata dict
        metadata = {
            "run_dir": str(path_obj.parent),
            "filename": path_obj.name,
        }
        
        # Extract dataset name from path
        dataset_match = re.search(r'(wiki(?:text|pedia)|openwebtext|shakespeare|books|c4|pile|code|cifar\d+)', 
                                  file_path, re.IGNORECASE)
        if dataset_match:
            metadata["dataset"] = dataset_match.group(1).lower()
        
        # Extract model size from path
        model_match = re.search(r'(gpt2?(?:-(?:small|medium|large|xl))?|llama-\d+[bm]|resnet\d+)', 
                               file_path, re.IGNORECASE)
        if model_match:
            metadata["model"] = model_match.group(1).lower()
        
        # Extract seed if present
        seed_match = re.search(r'seed[-_]?(\d+)', file_path, re.IGNORECASE)
        if seed_match:
            metadata["seed"] = int(seed_match.group(1))
            
        # Check if this is a RealignR run
        if "realignr" in file_path.lower():
            metadata["is_realignr"] = True
            
            # Try to extract RealignR version
            version_match = re.search(r'v(\d+(?:\.\d+)?)', file_path)
            if version_match:
                metadata["realignr_version"] = version_match.group(1)
        
        return metadata
    
    def load_config_if_exists(self, run_dir: str) -> Dict[str, Any]:
        """
        Load and parse config file if it exists in the run directory.
        
        Args:
            run_dir: Directory containing the run
            
        Returns:
            Dictionary of config parameters or empty dict if no config found
        """
        config = {}
        
        # Check for common config file names
        config_paths = [
            os.path.join(run_dir, name) for name in 
            ["config.json", "params.json", "hyperparams.json", "realignr_control.json", "realignr_phase.json"]
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        loaded_config = json.load(f)
                        
                    # Add prefix to avoid column name conflicts
                    prefixed_config = {f"config_{k}": v for k, v in loaded_config.items() 
                                      if isinstance(v, (int, float, str, bool))}
                    config.update(prefixed_config)
                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not parse config file {config_path}: {e}")
        
        return config
    
    def parse_csv_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse CSV file to extract relevant metrics.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary of extracted metrics
        """
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Initialize results dict with metadata
            results = self.extract_metadata_from_path(file_path)
            
            # Add config if exists
            config = self.load_config_if_exists(results["run_dir"])
            results.update(config)
            
            # Extract standard metrics if they exist in the CSV
            standard_metrics = [
                "train_loss", "val_loss", "test_loss", 
                "accuracy", "perplexity",
                "train_accuracy", "val_accuracy", "test_accuracy"
            ]
            
            for metric in standard_metrics:
                if metric in df.columns:
                    # Get the final value (last row)
                    results[f"final_{metric}"] = df[metric].iloc[-1]
                    
                    # Get the best value (min for loss/perplexity, max for accuracy)
                    if "loss" in metric or "perplexity" in metric:
                        results[f"best_{metric}"] = df[metric].min()
                    else:
                        results[f"best_{metric}"] = df[metric].max()
            
            # Look for ARP-specific metrics
            arp_metrics = ["G_mean", "alpha", "mu", "CPR_trigger", "drift"]
            
            for metric in arp_metrics:
                if metric in df.columns:
                    results[f"final_{metric}"] = df[metric].iloc[-1]
                    results[f"mean_{metric}"] = df[metric].mean()
                    results[f"max_{metric}"] = df[metric].max()
            
            # Extract run duration info
            if "step" in df.columns:
                results["total_steps"] = df["step"].max()
            elif "epoch" in df.columns:
                results["total_epochs"] = df["epoch"].max()
                
            # Try to get timestamp information
            try:
                results["run_date"] = datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d")
            except:
                pass
                
            return results
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {"file_path": file_path, "error": str(e)}
    
    def compile_leaderboard(self) -> pd.DataFrame:
        """
        Compile all results into a leaderboard DataFrame.
        
        Returns:
            DataFrame with all run metrics and metadata
        """
        # Scan for files
        csv_files = self.scan_directories()
        
        if not csv_files:
            print("No metric files found. Check your directory patterns.")
            return pd.DataFrame()
        
        # Parse each file
        all_results = []
        for file_path in csv_files:
            results = self.parse_csv_file(file_path)
            all_results.append(results)
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Handle primary metric for ranking
        if self.primary_metric in df.columns:
            rank_metric = self.primary_metric
        elif f"best_{self.primary_metric}" in df.columns:
            rank_metric = f"best_{self.primary_metric}"
        elif f"final_{self.primary_metric}" in df.columns:
            rank_metric = f"final_{self.primary_metric}"
        else:
            print(f"Warning: Primary metric '{self.primary_metric}' not found in results.")
            rank_metric = None
            
        # Compute ranks if primary metric exists
        if rank_metric and rank_metric in df.columns:
            # Sort by primary metric (ascending if lower is better, else descending)
            df = df.sort_values(by=rank_metric, ascending=self.lower_is_better)
            
            # Add rank column
            df["rank"] = range(1, len(df) + 1)
            
            # Add delta from best
            best_value = df[rank_metric].iloc[0]
            df["delta_from_best"] = df[rank_metric] - best_value
            if not self.lower_is_better:
                df["delta_from_best"] = -df["delta_from_best"]
        
        # Store the DataFrame
        self.df = df
        
        # Compute summary statistics
        self.compute_statistics()
        
        return df
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics about the leaderboard.
        
        Returns:
            Dictionary of summary statistics
        """
        if self.df is None or len(self.df) == 0:
            self.stats = {}
            return self.stats
            
        stats = {
            "total_runs": len(self.df),
        }
        
        # Count by dataset
        if "dataset" in self.df.columns:
            dataset_counts = self.df["dataset"].value_counts().to_dict()
            stats["dataset_counts"] = dataset_counts
        
        # Count by model
        if "model" in self.df.columns:
            model_counts = self.df["model"].value_counts().to_dict()
            stats["model_counts"] = model_counts
            
        # Count RealignR vs non-RealignR
        if "is_realignr" in self.df.columns:
            realignr_count = self.df["is_realignr"].sum()
            stats["realignr_count"] = realignr_count
            stats["non_realignr_count"] = len(self.df) - realignr_count
            
        # Get best run info
        if "rank" in self.df.columns:
            best_run = self.df[self.df["rank"] == 1].iloc[0].to_dict()
            stats["best_run"] = {k: v for k, v in best_run.items() 
                               if k not in ["run_dir", "filename"] and pd.notna(v)}
        
        self.stats = stats
        return stats
    
    def to_csv(self, output_path: str = "leaderboard.csv") -> str:
        """
        Save leaderboard to CSV file.
        
        Args:
            output_path: Path to save the CSV file
            
        Returns:
            Path to the saved file
        """
        if self.df is None:
            self.compile_leaderboard()
            
        if len(self.df) == 0:
            print("No data to save.")
            return None
            
        self.df.to_csv(output_path, index=False)
        print(f"Saved leaderboard to {output_path}")
        return output_path
    
    def to_json(self, output_path: str = "leaderboard.json") -> str:
        """
        Save leaderboard to JSON file.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            Path to the saved file
        """
        if self.df is None:
            self.compile_leaderboard()
            
        if len(self.df) == 0:
            print("No data to save.")
            return None
        
        # Convert DataFrame to JSON records
        leaderboard_data = {
            "runs": self.df.to_dict(orient="records"),
            "stats": self.stats,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(output_path, 'w') as f:
            json.dump(leaderboard_data, f, indent=2)
            
        print(f"Saved leaderboard to {output_path}")
        return output_path
    
    def to_markdown(self, output_path: Optional[str] = None, 
                   columns: List[str] = None) -> str:
        """
        Generate a markdown table from the leaderboard.
        
        Args:
            output_path: Path to save the markdown table (if None, just returns the string)
            columns: List of columns to include (if None, uses a default set)
            
        Returns:
            Markdown table as string
        """
        if self.df is None:
            self.compile_leaderboard()
            
        if len(self.df) == 0:
            return "No data available for markdown table."
        
        # Default columns if none specified
        if columns is None:
            # Try to find sensible default columns that exist
            possible_columns = [
                "rank", "run_dir", "dataset", "model", 
                f"best_{self.primary_metric}", f"final_{self.primary_metric}",
                "delta_from_best", "final_accuracy", "best_accuracy", 
                "final_G_mean", "total_steps", "total_epochs"
            ]
            
            columns = [col for col in possible_columns if col in self.df.columns]
            
            # Limit to a reasonable number of columns
            if len(columns) > 10:
                columns = columns[:10]
        
        # Generate markdown table
        md_table = "| " + " | ".join(columns) + " |\n"
        md_table += "| " + " | ".join(["---" for _ in columns]) + " |\n"
        
        # Add rows
        for _, row in self.df.iterrows():
            row_values = []
            for col in columns:
                if col in row:
                    val = row[col]
                    # Format numbers nicely
                    if isinstance(val, float):
                        if abs(val) < 0.01:
                            formatted = f"{val:.6f}"
                        else:
                            formatted = f"{val:.4f}"
                    else:
                        formatted = str(val)
                    
                    # Truncate long strings
                    if len(formatted) > 20 and col in ["run_dir", "filename"]:
                        formatted = "..." + formatted[-17:]
                        
                    row_values.append(formatted)
                else:
                    row_values.append("")
                    
            md_table += "| " + " | ".join(row_values) + " |\n"
        
        # Add summary stats
        if self.stats:
            md_table += f"\n**Total Runs**: {self.stats.get('total_runs', 0)}\n"
            
            if "dataset_counts" in self.stats:
                md_table += "\n**Datasets**:\n"
                for dataset, count in self.stats["dataset_counts"].items():
                    md_table += f"- {dataset}: {count} runs\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(md_table)
            print(f"Saved markdown table to {output_path}")
        
        return md_table
    
    def print_summary(self) -> None:
        """Print a summary of the leaderboard to the console."""
        if self.df is None:
            self.compile_leaderboard()
            
        if len(self.df) == 0:
            print("No data available.")
            return
        
        print("\n" + "="*80)
        print(f"REALIGNR LEADERBOARD SUMMARY ({len(self.df)} runs)")
        print("="*80)
        
        # Show top 5 runs
        print("\nTOP PERFORMERS:")
        print("-"*80)
        
        top_columns = ["rank", "dataset", "model", 
                       f"best_{self.primary_metric}", "delta_from_best"]
        top_columns = [col for col in top_columns if col in self.df.columns]
        
        if "run_dir" in self.df.columns:
            top_columns.append("run_dir")
            
        top_df = self.df.head(5)[top_columns]
        print(top_df.to_string(index=False))
        
        # Show stats
        if self.stats:
            print("\nSTATISTICS:")
            print("-"*80)
            
            if "dataset_counts" in self.stats:
                print("Datasets:")
                for dataset, count in self.stats["dataset_counts"].items():
                    print(f"  - {dataset}: {count} runs")
                    
            if "model_counts" in self.stats:
                print("\nModels:")
                for model, count in self.stats["model_counts"].items():
                    print(f"  - {model}: {count} runs")
                    
            if "realignr_count" in self.stats:
                print(f"\nRealignR runs: {self.stats['realignr_count']}")
                print(f"Non-RealignR runs: {self.stats['non_realignr_count']}")
        
        print("\n" + "="*80)


def main():
    """CLI entrypoint for the leaderboard autoloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RealignR Leaderboard Autoloader")
    parser.add_argument("--dirs", nargs="+", default=["results", "runs"],
                       help="Base directories to scan")
    parser.add_argument("--metric", default="val_loss",
                       help="Primary metric for ranking")
    parser.add_argument("--higher-better", action="store_true",
                       help="Set if higher values of metric are better")
    parser.add_argument("--csv", default="leaderboard.csv",
                       help="Path to save CSV output")
    parser.add_argument("--json", default="leaderboard.json",
                       help="Path to save JSON output")
    parser.add_argument("--markdown", default="leaderboard.md",
                       help="Path to save Markdown output")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save any files, just print summary")
    
    args = parser.parse_args()
    
    # Create leaderboard
    leaderboard = LeaderboardAutoloader(
        base_dirs=args.dirs,
        primary_metric=args.metric,
        lower_is_better=not args.higher_better
    )
    
    # Compile and show summary
    leaderboard.compile_leaderboard()
    leaderboard.print_summary()
    
    # Save outputs if requested
    if not args.no_save:
        leaderboard.to_csv(args.csv)
        leaderboard.to_json(args.json)
        leaderboard.to_markdown(args.markdown)
    
    return leaderboard


if __name__ == "__main__":
    main()
