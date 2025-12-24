import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional
import numpy as np

from .config import Config


def get_transforms(mode: str = 'train') -> transforms.Compose:
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomRotation(Config.ROTATION_DEGREES),
            transforms.RandomHorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
            transforms.RandomVerticalFlip(p=Config.VERTICAL_FLIP_PROB),
            transforms.ColorJitter(
                brightness=Config.COLOR_JITTER_BRIGHTNESS,
                contrast=Config.COLOR_JITTER_CONTRAST,
                saturation=Config.COLOR_JITTER_SATURATION
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD)
        ])

    elif mode in ['val', 'test', 'inference']:
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD)
        ])

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', 'test', or 'inference'")

    return transform


def preprocess_image(image: Image.Image, mode: str = 'inference') -> torch.Tensor:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = get_transforms(mode=mode)
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)

    return tensor


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = tensor.clone()

    mean = torch.tensor(Config.IMG_MEAN).view(3, 1, 1)
    std = torch.tensor(Config.IMG_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    image_np = tensor.cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)

    return image_np


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image_np = denormalize_image(tensor)
    return Image.fromarray(image_np)


def resize_image(image: Image.Image, size: Tuple[int, int] = None) -> Image.Image:
    if size is None:
        size = (Config.IMG_SIZE, Config.IMG_SIZE)

    return image.resize(size, Image.Resampling.LANCZOS)


def get_image_stats(image: Image.Image) -> dict:
    image_array = np.array(image)

    stats = {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'channels': len(image.getbands()),
        'mean': image_array.mean(axis=(0, 1)).tolist() if image_array.ndim == 3 else [image_array.mean()],
        'std': image_array.std(axis=(0, 1)).tolist() if image_array.ndim == 3 else [image_array.std()],
        'min': image_array.min(),
        'max': image_array.max()
    }

    return stats
