import math

import anyfig
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.data import setup_dataloaders
from src.models.model import get_model
from src.logger import Logger
from src.evaluation.validator import Validator
from src.evaluation.metrics import setup_metrics
from settings import configs
from src.utils.meta_utils import ProgressbarWrapper as Progressbar
from src.utils.meta_utils import speed_up_cuda
import src.utils.setup_utils as setup_utils


def train(config):
  speed_up_cuda()
  dataloaders = setup_dataloaders()
  metrics = setup_metrics()

  logger = Logger()
  model = get_model(config)
  validator = Validator(config)
  optimizer = torch.optim.Adam(model.parameters(), lr=config.start_lr)
  lr_scheduler = CosineAnnealingLR(optimizer,
                                   T_max=config.optim_steps,
                                   eta_min=config.end_lr)
  loss_fn = nn.CrossEntropyLoss()
  mixed_precision = config.mixed_precision and model.device != 'cpu'
  scaler = GradScaler(enabled=mixed_precision)

  # Init progressbar
  n_batches = len(dataloaders.train)
  n_epochs = math.ceil(config.optim_steps / n_batches)
  progressbar = Progressbar(n_epochs, n_batches)

  # Init variables
  optim_steps = 0
  val_freq = config.validation_freq

  # Training loop
  for epoch in progressbar(range(1, n_epochs + 1)):
    for batch_i, data in enumerate(dataloaders.train, 1):
      progressbar.update(epoch, batch_i)
      inputs, labels = data
      labels = labels.to(model.device)

      # Validation
      # if optim_steps % val_freq == 0:
      #   validator.validate(model, dataloaders.val, optim_steps)

      # Forward pass
      with autocast(mixed_precision):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

      # Backward pass
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      optim_steps += 1

      # Decrease learning rate
      lr_scheduler.step()

      # Log
      accuracy = metrics['accuracy'](outputs, labels)
      logger.log_accuracy(accuracy.item(), optim_steps)


if __name__ == '__main__':
  config = anyfig.init_config(default_config=configs.TrainLaptop)
  print(config)  # Remove if you dont want to see config at start
  print('\n{}\n'.format(config.misc.save_comment))
  setup_utils.setup(config.misc)
  train(config)
