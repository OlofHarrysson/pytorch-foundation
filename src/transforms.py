import imgaug.augmenters as iaa
import numpy as np
from anyfig import global_cfg
from torchvision import transforms


def get_train_augmenter():
    return Augmenter()


def get_val_augmenter():
    return get_train_augmenter()


class Augmenter:
    def __init__(self):
        img_size = global_cfg.input_size

        self.augmentations = [
            iaa.Resize({"height": img_size, "width": img_size}),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]

    def __call__(self, img, start=0, end=None):
        """Applies the transformations to the image

        Keyword Arguments:
        start {int} -- start index of transforms
        end {int} -- end index of transforms. None to include all. -1 to skip last transform
        """
        augmentations = self.augmentations[start:end]
        assert augmentations
        x = np.array(img)
        for augmentation in augmentations:
            if isinstance(augmentation, iaa.Augmenter):
                x = augmentation.augment_image(x)
            else:
                x = augmentation(x)
        return x
