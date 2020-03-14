from torchvision import transforms
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from anyfig import cfg


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
    im_size = cfg().input_size
    self.seq = iaa.Resize({"height": im_size, "width": im_size})

  def __call__(self, im):
    return self.seq.augment_image(np.array(im))
