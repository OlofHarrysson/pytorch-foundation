from torchvision import datasets
from torch.utils.data import DataLoader
from collections import namedtuple
from anyfig import cfg

from ..transforms import get_train_transforms, get_val_transforms
from ..utils.meta_utils import get_project_root


def setup_dataloaders():
  dataloaders = namedtuple('Dataloaders', ['train', 'val'])
  return dataloaders(train=setup_trainloader(), val=setup_valloader())


def setup_trainloader():
  transforms = get_train_transforms()
  dataset_dir = get_project_root() / 'datasets'
  dataset = MyCifar10(dataset_dir, transforms, train=True)
  return DataLoader(dataset,
                    batch_size=cfg().batch_size,
                    num_workers=cfg().num_workers,
                    shuffle=True)


def setup_valloader():
  transforms = get_val_transforms()
  dataset_dir = get_project_root() / 'datasets'
  dataset = MyCifar10(dataset_dir, transforms, train=False)
  return DataLoader(dataset,
                    batch_size=cfg().batch_size,
                    num_workers=cfg().num_workers)


class MyCifar10(datasets.CIFAR10):
  def __init__(self, path, transforms, train=True):
    super().__init__(path, train, download=True)
    self.transforms = transforms

  def __getitem__(self, index):
    im, label = super().__getitem__(index)
    return self.transforms(im), label
