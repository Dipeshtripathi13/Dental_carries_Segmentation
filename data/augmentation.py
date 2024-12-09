import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(config):
    return A.Compose([
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
        A.HorizontalFlip(p=config.AUG_PROB),
        A.Rotate(limit=config.ROTATE_LIMIT, p=config.AUG_PROB),
        A.RandomBrightnessContrast(
            brightness_limit=config.BRIGHTNESS_LIMIT,
            contrast_limit=config.CONTRAST_LIMIT,
            p=config.AUG_PROB
        ),
        A.GaussNoise(p=config.AUG_PROB),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_validation_augmentation(config):
    return A.Compose([
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
        A.Normalize(),
        ToTensorV2(),
    ])
