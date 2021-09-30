from collections import namedtuple

from anyfig import global_cfg
from torch.utils.data import DataLoader
from torchvision import datasets

from ..transforms import get_train_augmenter, get_val_augmenter
from ..utils.setup_utils import get_project_root


def setup_dataloaders():
    dataloaders = namedtuple("Dataloaders", ["train", "val"])
    return dataloaders(train=setup_trainloader(), val=setup_valloader())


def setup_trainloader():
    augmenter = get_train_augmenter()
    dataset_dir = get_project_root() / "datasets"
    dataset = MyCifar10(dataset_dir, augmenter, train=True)
    return DataLoader(
        dataset,
        batch_size=global_cfg.batch_size,
        num_workers=global_cfg.num_workers,
        shuffle=True,
    )


def setup_valloader():
    augmenter = get_val_augmenter()
    dataset_dir = get_project_root() / "datasets"
    dataset = MyCifar10(dataset_dir, augmenter, train=False)
    return DataLoader(
        dataset,
        batch_size=global_cfg.batch_size,
        num_workers=global_cfg.num_workers,
    )


class MyCifar10(datasets.CIFAR10):
    def __init__(self, path, transforms, train=True):
        super().__init__(path, train, download=True)
        self.transforms = transforms

    # def __len__(self):
    #     return 300  # TODO

    def __getitem__(self, index):
        im, label = super().__getitem__(index)
        return self.transforms(im), label
