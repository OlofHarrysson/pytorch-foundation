from abc import ABC, abstractclassmethod
import torch


def setup_metrics():
  return dict(accuracy=Accuracy(smoothing=0.9))


class Smoother():
  ''' Smooth value with EMA '''
  def __init__(self, smoothing):
    assert 0 <= smoothing <= 1, f'Expected value in range [0,1], was {smoothing}'
    self.smoothing = smoothing
    self.value = 0

  def __call__(self, new_value):
    s = self.smoothing
    self.value = s * self.value + (1 - s) * new_value
    return self.value


class MetricBase(ABC):
  def __call__(self, *args, **kwargs):
    return self.calc(*args, **kwargs)

  @abstractclassmethod
  def calc(self):
    ...


class Accuracy(MetricBase):
  def __init__(self, smoothing=0):
    self.smoother = Smoother(smoothing)

  def calc(self, outputs, labels):
    _, preds = torch.max(outputs, 1)
    accuracy = torch.sum(preds == labels, dtype=float) / len(preds)
    return self.smoother(accuracy)
