import argparse
import random
import numpy as np
import torch
import json
from git import Repo
from git.exc import InvalidGitRepositoryError

from .config_utils import choose_config
from .meta_utils import get_project_root, get_save_dir


def setup():
  config_str = parse_args()
  config = choose_config(config_str)
  print(config)
  print('\n{}\n'.format(config.save_comment))
  seed_program(config.seed)

  if config.save_experiment:
    save_experiment_info(config)
  return config


def parse_args():
  p = argparse.ArgumentParser()

  p.add_argument('--config',
                 type=str,
                 default='Cookie',
                 help='What config class to choose')

  args, unknown = p.parse_known_args()
  return args.config


def seed_program(seed=0):
  ''' Seed for reproducability '''
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  # torch.backends.cudnn.deterministic = True # You can add this


def save_experiment_info(config):
  ''' Saves experiment info in a timestamped directory '''
  save_dir = get_save_dir(config)
  save_dir.mkdir(exist_ok=True, parents=True)
  save_git_info(save_dir, config.start_time)
  save_config(config, save_dir)


def save_git_info(save_dir, timestamp):
  ''' Outputs a file with git info which can be used to track what code ran.
  It gets the latest commit hash and worktree difference against that commit '''
  project_dir = get_project_root()
  try:
    repo = Repo(project_dir)
    message = repo.head.commit.message
    hash_ = repo.head.object.hexsha
    worktree = repo.head.commit.tree
    diff = repo.git.diff(worktree)
  except InvalidGitRepositoryError as e:
    print("\nProject isn't tracked by git. Skipping saving git info\n")
    return

  git_info = {
    'latest_commit_message': message,
    'latest_commit_hexsha_hash': hash_,
    'experiment_start_time': timestamp,
    'diff_from_hexsha_hash': diff
  }

  with open(save_dir / 'git_info.txt', 'w') as f:
    for k, v in git_info.items():
      f.write('\n'.join([k, v]))
      f.write('\n' * 3)


def save_config(config, save_dir):
  with open(save_dir / 'config.json', 'w') as outfile:
    json.dump(config.get_parameters(), outfile, indent=2)