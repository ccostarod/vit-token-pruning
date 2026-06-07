import numpy as np
import cv2
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

def get_train_transforms(
    image_size: int = 224,
    use_augmentation: bool = False,
    random_erasing_p: float = 0.0,
):
    if not use_augmentation:
        return get_eval_transforms(image_size)

    transforms = [
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
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        ),
        A.ColorJitter(
            brightness=(0.9, 1.1),
            contrast=(0.9, 1.1),
            saturation=(0.9, 1.1),
            hue=(-0.03, 0.03),
            p=0.3,
        ),
    ]

    if random_erasing_p > 0.0:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(0.02, 0.2),
                hole_width_range=(0.02, 0.2),
                fill="random_uniform",
                p=random_erasing_p,
            )
        )

    transforms.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return AlbumentationsTransform(
        A.Compose(transforms)
    )


def get_eval_transforms(image_size: int = 224):
    return AlbumentationsTransform(
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    )
