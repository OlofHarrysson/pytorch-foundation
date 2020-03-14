import torch
import torch.nn as nn

class Validator():
  def __init__(self, config):
    pass

  def validate(self, model, val_loader, step):
    for batch_i, data in enumerate(val_loader, 1):
      inputs, labels = data
      outputs = model.predict(inputs)
      loss, accuracy = model.calc_loss(outputs, labels, accuracy=True)
      print(f'Validation accuracy: {accuracy}')