"""
RealignR Leaderboard CLI

A command-line script to run the leaderboard autoloader and dashboard generator together.
"""

import os
import argparse
from leaderboard_autoloader import LeaderboardAutoloader
from leaderboard_dashboard import LeaderboardDashboard

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RealignR Leaderboard Generator")
    
    # Common arguments
    parser.add_argument("--dirs", nargs="+", default=["runs", "results"],
                      help="Directories to scan for results (default: runs results)")
    parser.add_argument("--metric", type=str, default="val_loss",
                      help="Primary metric to sort by (default: val_loss)")
    parser.add_argument("--higher-better", action="store_true",
                      help="Set if higher values of metric are better (default: False)")
    
    # Leaderboard arguments
    parser.add_argument("--csv", type=str, default="leaderboard.csv",
                      help="Path to save CSV output (default: leaderboard.csv)")
    parser.add_argument("--json", type=str, default="leaderboard.json",
                      help="Path to save JSON output (default: leaderboard.json)")
    parser.add_argument("--markdown", type=str, default="leaderboard.md",
                      help="Path to save Markdown output (default: leaderboard.md)")
    
    # Dashboard arguments
    parser.add_argument("--output-dir", type=str, default="dashboard",
                      help="Directory to save dashboard outputs (default: dashboard)")
    parser.add_argument("--dashboard-only", action="store_true",
                      help="Only generate dashboard from existing leaderboard CSV/JSON")
    parser.add_argument("--leaderboard-only", action="store_true",
                      help="Only generate leaderboard, skip dashboard")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI script."""
    args = parse_args()
    
    if args.dashboard_only:
        # Load leaderboard from existing file
        print(f"Loading leaderboard from {args.csv if os.path.exists(args.csv) else args.json}")
        dashboard = LeaderboardDashboard(
            leaderboard_file=args.csv if os.path.exists(args.csv) else args.json,
            primary_metric=args.metric,
            lower_is_better=not args.higher_better,
            output_dir=args.output_dir
        )
        html_path = dashboard.generate_html_dashboard()
        print(f"Dashboard generated at: {html_path}")
        
    else:
        # Create leaderboard
        print(f"Scanning directories: {args.dirs}")
        loader = LeaderboardAutoloader(
            base_dirs=args.dirs,
            primary_metric=args.metric,
            lower_is_better=not args.higher_better
        )
        
        # Compile leaderboard and save outputs
        loader.compile_leaderboard()
        loader.print_summary()
        
        loader.to_csv(args.csv)
        loader.to_json(args.json)
        loader.to_markdown(args.markdown)
        
        # Generate dashboard if requested
        if not args.leaderboard_only:
            print("Generating dashboard...")
            dashboard = LeaderboardDashboard(
                leaderboard_loader=loader,
                primary_metric=args.metric,
                lower_is_better=not args.higher_better,
                output_dir=args.output_dir
            )
            html_path = dashboard.generate_html_dashboard()
            print(f"Dashboard generated at: {html_path}")
    
if __name__ == "__main__":
    main()
