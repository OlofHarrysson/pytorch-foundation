import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from anyfig import global_cfg
from src.evaluation.metrics import setup_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR


def setup_model():
    return MyModel()


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.metrics = setup_metrics()
        self.learning_rate = global_cfg.start_lr
        self.backbone = models.resnet18(pretrained=global_cfg.pretrained)
        n_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(n_features, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs):
        return self.backbone(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("training/loss", loss, on_step=True)

        metric_scores = self.metrics.train(preds, y)
        self.log_dict(metric_scores, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("validation/loss", loss, on_epoch=True)

        metric_scores = self.metrics.val(preds, y)
        self.log_dict(metric_scores, on_epoch=True)

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
