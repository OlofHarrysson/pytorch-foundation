import anyfig
import pyjokes
import random
from datetime import datetime


@anyfig.config_class
class MiscConfig():
  # ~~~~~~~~~~~~~~ General Parameters ~~~~~~~~~~~~~~
  def __init__(self):
    super().__init__()

    # Creates directory. Saves config & git info
    self.save_experiment: bool = False

    # An optional comment to differentiate this run from others
    self.save_comment: str = pyjokes.get_joke()

    # Start time to keep track of when the experiment was run
    self.start_time: str = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # Seed for reproducability
    self.seed: int = random.randint(0, 2**31)

    # Decides if logger should be active
    self.log_data: bool = False


@anyfig.config_class
class TrainingConfig():
  # ~~~~~~~~~~~~~~ Training Parameters ~~~~~~~~~~~~~~
  def __init__(self):

    # Use GPU. Set to False to only use CPU
    self.use_gpu: bool = True

    # Number of threads to use in data loading
    self.num_workers: int = 0

    # Number of update steps to train
    self.optim_steps: int = 10000

    # Number of optimization steps between validation
    self.validation_freq: int = 500

    # Start and end learning rate for the scheduler
    self.start_lr: float = 1e-3
    self.end_lr: float = 1e-4

    # Batch size going into the network
    self.batch_size: int = 32

    # Size for image that is fed into the network
    self.input_size = 64

    # Use a pretrained network
    self.pretrained: bool = False

    # Misc configs
    self.misc = MiscConfig()


@anyfig.config_class
class TrainLaptop(TrainingConfig):
  def __init__(self):
    super().__init__()
    ''' Change default parameters here. Like this
    self.seed = 666            ____
      ________________________/ O  \___/  <--- Python <3
     <_#_#_#_#_#_#_#_#_#_#_#_#_____/   \
    '''
    self.use_gpu = False
    self.misc.log_data = True
    # self.misc.save_experiment: bool = True


@anyfig.config_class
class TrainMegaMachine(TrainingConfig):
  def __init__(self):
    super().__init__()
    self.batch_size = self.batch_size * 4
