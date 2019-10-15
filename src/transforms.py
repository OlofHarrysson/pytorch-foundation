from torchvision import transforms
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image


def get_train_transforms():
  transformer = Transformer()

  return transforms.Compose([
    transformer,
    transforms.ToTensor(),
  ])


def get_val_transforms():
  return get_train_transforms()


class Transformer():
  def __init__(self):
    self.seq = iaa.Resize({"height": 64, "width": 64})

  def __call__(self, im):
    augmented_im = self.seq.augment_image(np.array(im))
    return Image.fromarray(augmented_im)
