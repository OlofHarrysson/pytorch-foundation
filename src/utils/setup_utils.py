from pathlib import Path

import anyfig
from git import Repo
from git.exc import InvalidGitRepositoryError


def setup_experiment(config, save_dir):
    if config.misc.save_experiment:
        save_experiment_info(config, save_dir)


def save_experiment_info(config, save_dir):
    """Saves experiment info in a timestamped directory"""
    save_dir.mkdir(exist_ok=True, parents=True)
    save_git_info(save_dir, config.misc.start_time)
    anyfig.save_config(config, save_dir / "config.cfg")


def save_git_info(save_dir, timestamp):
    """Outputs a file with git info which can be used to track what code ran.
    It gets the latest commit hash and worktree difference against that commit"""
    project_dir = get_project_root()
    try:
        repo = Repo(project_dir)
        message = repo.head.commit.message
        hash_ = repo.head.object.hexsha
        worktree = repo.head.commit.tree
        diff = repo.git.diff(worktree)
    except InvalidGitRepositoryError:
        print("\nProject isn't tracked by git. Skipping saving git info\n")
        return

    git_info = {
        "latest_commit_message": message,
        "latest_commit_hexsha_hash": hash_,
        "experiment_start_time": timestamp,
        "diff_from_hexsha_hash": diff,
    }

    with open(save_dir / "git_info.txt", "w") as f:
        for k, v in git_info.items():
            f.write("\n".join([k, v]))
            f.write("\n" * 3)


def get_project_root():
    """Returns project root folder"""
    return Path(__file__).parents[2]
