import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path


def get_model(config):
  model = MyModel(config)
  model = model.to(model.device)
  return model


class MyModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cpu' if config.gpu < 0 else torch.device('cuda', config.gpu)

    self.backbone = models.resnet18(pretrained=config.pretrained)
    n_features = self.backbone.fc.in_features
    self.backbone.fc = nn.Linear(n_features, 10)

  def forward(self, inputs):
    inputs = inputs.to(self.device)
    return self.backbone(inputs)

  def predict(self, inputs):
    with torch.no_grad():
      return self(inputs)

  def save(self, path):
    path = Path(path)
    err_msg = f"Expected path that ends with '.pt' or '.pth' but was '{path}'"
    assert path.suffix in ['.pt', '.pth'], err_msg
    path.parent.mkdir(exist_ok=True)
    print("Saving Weights @ " + str(path))
    torch.save(self.state_dict(), path)

  def load(self, path):
    print('Loading weights from {}'.format(path))
    weights = torch.load(path, map_location='cpu')
    self.load_state_dict(weights, strict=False)
    self.to(self.device)
