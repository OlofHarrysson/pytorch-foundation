import anyfig
import pyjokes, random
from datetime import datetime as dtime


class MiscConfig(anyfig.MasterConfig):
  # ~~~~~~~~~~~~~~ General Parameters ~~~~~~~~~~~~~~
  def __init__(self):
    super().__init__()

    # Saves config & git diffs
    self.save_experiment: bool = False

    # An optional comment to differentiate this run from others
    self.save_comment: str = pyjokes.get_joke()

    # Start time to keep track of when the experiment was run
    self.start_time: str = dtime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Seed to create reproducable training results
    self.seed: int = random.randint(0, 2**32 - 1)

    # Decides if logger should be active
    self.log_data: bool = False


class TrainingConfig(anyfig.MasterConfig):
  def __init__(self):
    super().__init__()
    # Use GPU. Set to False to only use CPU
    self.use_gpu: bool = True

    # Threads to use in data loading
    self.num_workers: int = 0

    # Batch size going into the network
    self.batch_size: int = 32

    # Using a pretrained network
    self.pretrained: bool = False

    # Start and end learning rate for the scheduler
    self.start_lr: float = 1e-3
    self.end_lr: float = 1e-4

    # For how many steps to train
    self.optim_steps: int = 10000

    # How many optimization steps before validation
    self.validation_freq: int = 100


@anyfig.config_class
class Cookie(TrainingConfig, MiscConfig):
  def __init__(self):
    super().__init__()
    ''' Change default parameters here. Like this
    self.seed = 666            ____
      ________________________/ O  \___/  <--- Python <3
     <_#_#_#_#_#_#_#_#_#_#_#_#_____/   \
    '''
    self.use_gpu = False