"""Git sync — version control the knowledge base.

Syncs the corpus source directory with a git remote.
Enables collaboration and backup of knowledge bases.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyncResult:
    action: str  # "pulled", "pushed", "up-to-date", "initialized"
    message: str
    changed_files: int


class GitSync:
    @staticmethod
    def is_git_repo(path: Path) -> bool:
        """Check if path is inside a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=str(path),
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def init(path: Path) -> SyncResult:
        """Initialize a git repo at path if not already one."""
        if GitSync.is_git_repo(path):
            return SyncResult("up-to-date", "Already a git repository", 0)
        subprocess.run(["git", "init"], cwd=str(path), capture_output=True, check=True)
        return SyncResult("initialized", f"Initialized git repository at {path}", 0)

    @staticmethod
    def status(path: Path) -> dict:
        """Get git status of the knowledge base."""
        if not GitSync.is_git_repo(path):
            return {"is_repo": False}
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(path),
            capture_output=True,
            text=True,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]

        # Get current branch
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(path),
            capture_output=True,
            text=True,
        )

        return {
            "is_repo": True,
            "branch": branch_result.stdout.strip(),
            "changed_files": len(lines),
            "changes": lines[:20],  # Cap at 20 for display
        }

    @staticmethod
    def commit_all(path: Path, message: str = "Update knowledge base") -> SyncResult:
        """Stage all changes and commit."""
        if not GitSync.is_git_repo(path):
            return SyncResult("error", "Not a git repository", 0)

        # Stage all
        subprocess.run(["git", "add", "-A"], cwd=str(path), capture_output=True)

        # Check if there's anything to commit
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(path),
            capture_output=True,
            text=True,
        )
        changed = len([l for l in status.stdout.strip().split("\n") if l.strip()])

        if changed == 0:
            return SyncResult("up-to-date", "Nothing to commit", 0)

        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "commit", "-m", message],
            cwd=str(path),
            capture_output=True,
            check=True,
        )
        return SyncResult("committed", f"Committed {changed} changes", changed)

    @staticmethod
    def pull(path: Path) -> SyncResult:
        """Pull latest changes from remote."""
        if not GitSync.is_git_repo(path):
            return SyncResult("error", "Not a git repository", 0)
        try:
            result = subprocess.run(
                ["git", "pull", "--rebase"],
                cwd=str(path),
                capture_output=True,
                text=True,
            )
            if "Already up to date" in result.stdout:
                return SyncResult("up-to-date", "Already up to date", 0)
            return SyncResult("pulled", result.stdout.strip(), 0)
        except subprocess.CalledProcessError as e:
            return SyncResult("error", f"Pull failed: {e.stderr}", 0)

    @staticmethod
    def push(path: Path) -> SyncResult:
        """Push changes to remote."""
        if not GitSync.is_git_repo(path):
            return SyncResult("error", "Not a git repository", 0)
        try:
            result = subprocess.run(
                ["git", "push"],
                cwd=str(path),
                capture_output=True,
                text=True,
            )
            return SyncResult("pushed", "Pushed to remote", 0)
        except subprocess.CalledProcessError as e:
            return SyncResult("error", f"Push failed: {e.stderr}", 0)
