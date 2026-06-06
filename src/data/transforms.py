import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        image = np.asarray(image.convert("RGB"))
        return self.transform(image=image)["image"]


def get_train_transforms(image_size: int = 224, use_augmentation: bool = False):
    if not use_augmentation:
        return get_eval_transforms(image_size)

    return AlbumentationsTransform(
        A.Compose([
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                border_mode=4,
                p=0.3,
            ),
            A.ColorJitter(
                brightness_range=(0.9, 1.1),
                contrast_range=(0.9, 1.1),
                saturation_range=(0.9, 1.1),
                hue_range=(-0.03, 0.03),
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    )


def get_eval_transforms(image_size: int = 224):
    return AlbumentationsTransform(
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    )
