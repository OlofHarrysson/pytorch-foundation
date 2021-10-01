from pathlib import Path

import anyfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from find_learning_rate import find_best_learning_rate
from settings import configs
from src.data.data import setup_dataloaders
from src.evaluation.metrics import setup_metrics
from src.models.model import setup_model
from src.utils import setup_utils


def train(config, find_best_lr):
    # TODO check on GPU machine
    mixed_precision = config.mixed_precision and config.gpus is not None
    precision = 16 if mixed_precision else 32
    fast_dev_run = 10 if config.fast_dev_run else False
    logger = TensorBoardLogger(
        save_dir=setup_utils.get_project_root() / "output" / "trainings",
        name=config.misc.start_time,
        version="logs",
        default_hp_metric=False,
    )
    exp_dir = Path(logger.log_dir).parent
    setup_utils.setup(config, exp_dir)

    trainer = pl.Trainer(
        default_root_dir=exp_dir,
        precision=precision,
        fast_dev_run=fast_dev_run,
        logger=logger,
    )

    metrics = setup_metrics()
    model = setup_model(config, metrics)
    dataloaders = setup_dataloaders()
    if find_best_lr:
        find_best_learning_rate(trainer, model, dataloaders)
    else:
        trainer.fit(
            model,
            train_dataloaders=dataloaders.train,
            val_dataloaders=dataloaders.val,
        )


def main(find_best_lr=False):
    config = anyfig.init_config(default_config=configs.TrainLaptop)
    print(config)  # Remove if you dont want to see config at start
    print("\n{}\n".format(config.misc.save_comment))
    train(config, find_best_lr)


if __name__ == "__main__":
    main()
