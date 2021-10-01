import sys

import anyfig
import pytorch_lightning as pl

from settings import configs
from src.data.data import setup_dataloaders
from src.models.model import setup_model


def find_best_learning_rate():
    config = anyfig.init_config(default_config=configs.TrainLaptop)
    print(config)  # Remove if you dont want to see config at start

    model = setup_model()
    dataloaders = setup_dataloaders()
    trainer = pl.Trainer(gpus=config.gpus)
    lr_finder = trainer.tuner.lr_find(model, dataloaders.train)
    if lr_finder:
        fig = lr_finder.plot(suggest=True)
        fig.show()
        print(f"Suggested learning rate: {lr_finder.suggestion()}")
        input("Showing learning rate graph. Press Enter key to quit")
    sys.exit(0)


if __name__ == "__main__":
    find_best_learning_rate()
