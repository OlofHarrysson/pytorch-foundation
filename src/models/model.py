import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as models
from anyfig import global_cfg
from torch.optim.lr_scheduler import CosineAnnealingLR


def setup_model(config):
    return MyModel(config)


class MyModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.learning_rate = global_cfg.start_lr
        self.backbone = models.resnet18(pretrained=config.pretrained)
        n_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(n_features, 10)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, inputs):
        return self.backbone(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("training/loss", loss)

        self.train_acc(preds, y)
        self.log("training/accuracy", self.train_acc, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        self.valid_acc(preds, y)
        self.log("validation/accuracy", self.valid_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=global_cfg.optim_steps, eta_min=global_cfg.end_lr
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
        }
        return dict(optimizer=optimizer, lr_scheduler_config=lr_scheduler_config)
