import pyjokes, random
from datetime import datetime as dtime
from collections import OrderedDict
import pprint
from abc import ABC
from dataclasses import dataclass, FrozenInstanceError


@dataclass
class DefaultConfig(ABC):
  def __init__(self, config_str):
    # ~~~~~~~~~~~~~~ General Parameters ~~~~~~~~~~~~~~
    # The config name
    self.config = config_str

    # An optional comment to differentiate this run from others
    self.save_comment = pyjokes.get_joke()

    # Seed to create reproducable training results
    self.seed = random.randint(0, 2**32 - 1)

    # Start time to keep track of when the experiment was run
    self.start_time = dtime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Freezes the config after setup, turning it immutable
    self.freeze_config = True

    # Decides if logger should be active
    self.log_data = False

    # Use GPU. Set to False to only use CPU
    self.use_gpu = True

    # Threads to use in data loading
    self.num_workers = 0

    # Batch size going into the network
    self.batch_size = 32

    # Using a pretrained network
    self.pretrained = False

    # Start and end learning rate for the scheduler
    self.start_lr = 1e-3
    self.end_lr = 1e-4

    # For how many steps to train
    self.optim_steps = 10000

    # How often to validate
    self.validation_freq = 100

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