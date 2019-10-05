import progressbar as pbar
import random, torch
import numpy as np
from collections import deque

def seed_program(seed=0):
  ''' Seed for reproducability '''
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  # torch.backends.cudnn.deterministic = True # You can add this

class ProgressbarWrapper():
  def __init__(self, n_epochs, n_batches):
    self.text = pbar.FormatCustomText(
      'Epoch: %(epoch).d/%(n_epochs).d, Batch: %(batch)d/%(n_batches)d',
      dict(
        epoch=0,
        n_epochs=n_epochs,
        batch=0,
        n_batches=n_batches
      ),
    )

    self.bar = pbar.ProgressBar(widgets=[
        self.text, '    ',
        pbar.Timer(), '    ',
        pbar.AdaptiveETA(), '',
    ], redirect_stdout=True)

    self.bar.start()

  def __call__(self, down_you_go):
    return self.bar(down_you_go)

  def update(self, epoch, batch):
    self.text.update_mapping(epoch=epoch, batch=batch)
    self.bar.update()