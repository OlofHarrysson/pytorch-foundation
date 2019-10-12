import pyjokes, random
from datetime import datetime as dtime
from collections import OrderedDict
import pprint
from abc import ABC
from dataclasses import dataclass, FrozenInstanceError


@dataclass
class DefaultConfig(ABC):
  # ~~~~~~~~~~~~~~ General Parameters ~~~~~~~~~~~~~~
  # The config name
  config: str

  # An optional comment to differentiate this run from others
  save_comment: str = pyjokes.get_joke()

  # Seed to create reproducable training results
  seed: int = random.randint(0, 2**32 - 1)

  # Start time to keep track of when the experiment was run
  start_time: str = dtime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

  # Freezes the config after setup, turning it immutable
  freeze_config: bool = True

  # Decides if logger should be active
  log_data: bool = False

  # Use GPU. Set to False to only use CPU
  use_gpu: bool = True

  # Threads to use in data loading
  num_workers: int = 0

  # Batch size going into the network
  batch_size: int = 32

  # Using a pretrained network
  pretrained: bool = False

  # Start and end learning rate for the scheduler
  start_lr: float = 1e-3
  end_lr: float = 1e-4

  # For how many steps to train
  optim_steps: int = 10000

  # How often to validate
  validation_freq: int = 100

  def get_parameters(self):
    return OrderedDict(sorted(vars(self).items()))

  def __str__(self):
    return pprint.pformat(dict(self.get_parameters()))

  def freeze(self):
    ''' Freezes object, making it immutable '''
    def handler(self, name, value):
      err_msg = f"Cannot assign to field '{name}'. Config object is frozen. Change 'freeze_config' to False if you want a mutable config object"
      raise FrozenInstanceError(err_msg)

    setattr(DefaultConfig, '__setattr__', handler)


class Cookie(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    ''' Change default parameters here. Like this
    self.seed = 666          ____
      ________________________/ O  \___/  <--- Python <3
     <_#_#_#_#_#_#_#_#_#_#_#_#_____/   \
    '''
    self.use_gpu = False