import torch
import torch.nn as nn
import torchvision.models as models

def get_model(config):
  return MyModel(config)

class MyModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'
    self.loss_fn = nn.CrossEntropyLoss()

    self.backbone = models.resnet18(pretrained=config.pretrained)
    n_features = self.backbone.fc.in_features
    self.backbone.fc = nn.Linear(n_features, 10)

  def forward(self, inputs):
    inputs = inputs.to(self.device)
    return self.backbone(inputs)

  def predict(self, inputs):
    with torch.no_grad():
      return self(inputs)

  def calc_loss(self, outputs, labels, accuracy=False):
    labels = labels.to(self.device)

    _, preds = torch.max(outputs, 1)
    accuracy = torch.sum(preds == labels, dtype=float) / len(preds)

    return self.loss_fn(outputs, labels), accuracy

  def save(self, path):
    save_dir = Path(path).parent
    save_dir.mkdir(exist_ok=True, parents=True)
    print("Saving Weights @ " + path)
    torch.save(self.state_dict(), path)

  def load(self, path):
    print('Loading weights from {}'.format(path))
    weights = torch.load(path, map_location='cpu')
    self.load_state_dict(weights, strict=False)