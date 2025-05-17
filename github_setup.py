"""
GitHub Repository Setup Script

This script initializes a Git repository for the RealignR project and pushes it to GitHub.
Before running, make sure you:
1. Have Git installed
2. Have a GitHub account

The script is preconfigured to push to https://github.com/RDM3DC/realignerpreview.git

Usage:
    python github_setup.py
"""

import subprocess
import os
import sys

def run_command(command, description=None):
    """Run a shell command and print output."""
    if description:
        print(f"\n{description}...\n")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              stderr=subprocess.PIPE, stdout=subprocess.PIPE, 
                              text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Details: {e.stderr}")
        return False

def setup_github_repo():
    """Setup and push to GitHub repository."""
    # Initialize Git repository
    if not run_command("git init", "Initializing Git repository"):
        return False
    
    # Add all files
    if not run_command("git add .", "Adding files to repository"):
        return False
    
    # Commit
    if not run_command('git commit -m "Initial commit with RealignR optimizer and leaderboard system"', 
                     "Creating initial commit"):
        return False
    
    # Add remote
    github_url = "https://github.com/RDM3DC/realignerpreview.git"
    if not run_command(f"git remote add origin {github_url}", 
                    f"Adding GitHub remote ({github_url})"):
        return False
      # Push to GitHub
    print("\nReady to push to GitHub!")
    print("Note: You may be prompted for GitHub credentials.")
    
    push = input("\nPush to GitHub now? (y/n): ").strip().lower()
    if push == 'y':
        if not run_command("git push -u origin main", "Pushing to GitHub"):
            # Try master branch if main doesn't work
            run_command("git push -u origin master", "Trying alternate branch name")
    else:
        print("\nSkipping push to GitHub. You can push manually with: git push -u origin main")
    
    print("\nSetup complete!")
    print("Repository URL: https://github.com/RDM3DC/realignerpreview.git")
    
    return True

if __name__ == "__main__":
    # Run setup with preconfigured repository
    setup_github_repo()
