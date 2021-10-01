import random
from datetime import datetime

import anyfig
import pyjokes


@anyfig.config_class
class MiscConfig:
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
        self.seed: int = random.randint(0, 2 ** 31)

        # Decides if logger should be active
        self.log_data: bool = False


@anyfig.config_class
class TrainingConfig:
    # ~~~~~~~~~~~~~~ Training Parameters ~~~~~~~~~~~~~~
    def __init__(self):

        # Run the training in fast mode useful for debugging
        self.fast_dev_run: bool = False

        # The GPU device to use. Set the value to None to use the CPU
        self.gpus: int = 0

        # Runs the computations with mixed precision. Only works with GPUs enabled
        self.mixed_precision = True

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
        self.input_size = 32

        # Use a pretrained network
        self.pretrained: bool = False

        # Misc configs
        self.misc = MiscConfig()


@anyfig.config_class
class TrainLaptop(TrainingConfig):
    def __init__(self):
        super().__init__()
        """ Change default parameters here. Like this
        self.seed = 666            ____
         ________________________/ O   \___/  <--- Python <3
        <_#_#_#_#_#_#_#_#_#_#_#_#_____/    \
        """
        # self.fast_dev_run: bool = True
        self.misc.save_experiment: bool = False
        self.gpus: int = None
        # self.num_workers = 4
        self.batch_size = 8


@anyfig.config_class
class TrainMegaMachine(TrainingConfig):
    def __init__(self):
        super().__init__()
        self.batch_size = self.batch_size * 4
