from collections import namedtuple

import torchmetrics


def setup_metrics():
    metrics = namedtuple("Metrics", ["train", "val"])
    return metrics(train=setup_trainmetrics(), val=setup_valmetrics())


def setup_trainmetrics():
    return torchmetrics.MetricCollection(
        [
            torchmetrics.Accuracy(),
        ],
        prefix="training/",
    )


def setup_valmetrics():
    return torchmetrics.MetricCollection(
        [
            torchmetrics.Accuracy(),
            torchmetrics.F1(average="macro", num_classes=10),
        ],
        prefix="validation/",
    )
