from pathlib import Path
from mcq.sync.git import GitSync


def test_is_git_repo_false(tmp_path):
    assert GitSync.is_git_repo(tmp_path) is False


def test_init_creates_repo(tmp_path):
    result = GitSync.init(tmp_path)
    assert result.action == "initialized"
    assert GitSync.is_git_repo(tmp_path) is True


def test_init_idempotent(tmp_path):
    GitSync.init(tmp_path)
    result = GitSync.init(tmp_path)
    assert result.action == "up-to-date"


def test_status_not_repo(tmp_path):
    status = GitSync.status(tmp_path)
    assert status["is_repo"] is False


def test_status_clean_repo(tmp_path):
    GitSync.init(tmp_path)
    # Need at least one commit for status to work
    (tmp_path / "test.txt").write_text("hello\n")
    import subprocess
    subprocess.run(["git", "add", "-A"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-m", "init"], cwd=str(tmp_path), capture_output=True)
    status = GitSync.status(tmp_path)
    assert status["is_repo"] is True
    assert status["changed_files"] == 0


def test_commit_all(tmp_path):
    GitSync.init(tmp_path)
    (tmp_path / "note.md").write_text("# Note\n")
    result = GitSync.commit_all(tmp_path, "initial")
    assert result.action == "committed"
    assert result.changed_files > 0


def test_commit_nothing_to_commit(tmp_path):
    GitSync.init(tmp_path)
    (tmp_path / "note.md").write_text("# Note\n")
    GitSync.commit_all(tmp_path, "first")
    result = GitSync.commit_all(tmp_path, "second")
    assert result.action == "up-to-date"
