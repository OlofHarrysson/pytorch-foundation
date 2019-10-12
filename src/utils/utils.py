import progressbar as pbar
import random, torch
import numpy as np
from collections import deque
from git import Repo


def seed_program(seed=0):
  ''' Seed for reproducability '''
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  # torch.backends.cudnn.deterministic = True # You can add this


def save_experiment_info(experiment_dir):
  print(experiment_dir)
  git_dir = Repo('.', search_parent_directories=True)
  print(git_dir)
  print(git_dir.working_tree_dir)
  print(git_dir.index)
  t = git_dir.head.commit.tree
  print(git_dir.git.diff(t))
  qe


class ProgressbarWrapper():
  def __init__(self, n_epochs, n_batches):
    self.text = pbar.FormatCustomText(
      'Epoch: %(epoch).d/%(n_epochs).d, Batch: %(batch)d/%(n_batches)d',
      dict(epoch=0, n_epochs=n_epochs, batch=0, n_batches=n_batches),
    )

    self.bar = pbar.ProgressBar(widgets=[
      self.text,
      '    ',
      pbar.Timer(),
      '    ',
      pbar.AdaptiveETA(),
      '',
    ],
                                redirect_stdout=True)

    self.bar.start()

  def __call__(self, down_you_go):
    return self.bar(down_you_go)

  def update(self, epoch, batch):
    self.text.update_mapping(epoch=epoch, batch=batch)
    self.bar.update()