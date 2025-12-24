import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Callable
import os

from utils.preprocessing import get_transforms
from utils.config import Config


class SkinDiseaseDataset(datasets.ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None, mode: str = 'train'):
        if transform is None:
            transform = get_transforms(mode=mode)

        super().__init__(root=root, transform=transform)
        self.mode = mode

        print(f"✓ Loaded {mode} dataset from: {root}")
        print(f"  - Total samples: {len(self)}")
        print(f"  - Number of classes: {len(self.classes)}")

    def get_class_distribution(self) -> dict:
        class_counts = {}
        for _, class_idx in self.samples:
            class_name = self.classes[class_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return class_counts

    def get_sample_weights(self) -> torch.Tensor:
        class_counts = self.get_class_distribution()
        total_samples = len(self)

        class_weights = {
            class_name: total_samples / count
            for class_name, count in class_counts.items()
        }

        sample_weights = []
        for _, class_idx in self.samples:
            class_name = self.classes[class_idx]
            sample_weights.append(class_weights[class_name])

        return torch.DoubleTensor(sample_weights)


def create_dataloaders(train_dir: Optional[str] = None,
                      test_dir: Optional[str] = None,
                      batch_size: Optional[int] = None,
                      num_workers: Optional[int] = None,
                      pin_memory: Optional[bool] = None,
                      use_weighted_sampling: bool = False) -> Tuple[DataLoader, DataLoader]:
    train_dir = train_dir or str(Config.TRAIN_DIR)
    test_dir = test_dir or str(Config.TEST_DIR)
    batch_size = batch_size or Config.BATCH_SIZE
    num_workers = num_workers or Config.NUM_WORKERS
    pin_memory = pin_memory if pin_memory is not None else Config.PIN_MEMORY

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_dataset = SkinDiseaseDataset(
        root=train_dir,
        mode='train'
    )

    test_dataset = SkinDiseaseDataset(
        root=test_dir,
        mode='test'
    )

    train_sampler = None
    shuffle = Config.SHUFFLE_TRAIN

    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"\n✓ DataLoaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Weighted sampling: {use_weighted_sampling}")

    return train_loader, test_loader


def get_dataset_statistics(dataset: SkinDiseaseDataset) -> dict:
    class_dist = dataset.get_class_distribution()

    stats = {
        'total_samples': len(dataset),
        'num_classes': len(dataset.classes),
        'class_names': dataset.classes,
        'class_distribution': class_dist,
        'samples_per_class': {
            'min': min(class_dist.values()),
            'max': max(class_dist.values()),
            'mean': sum(class_dist.values()) / len(class_dist),
        }
    }

    return stats


def print_dataset_info(train_loader: DataLoader, test_loader: DataLoader):
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)

    train_dataset = train_loader.dataset
    train_stats = get_dataset_statistics(train_dataset)

    print(f"\n📊 Training Set:")
    print(f"  Total samples: {train_stats['total_samples']}")
    print(f"  Number of classes: {train_stats['num_classes']}")
    print(f"  Samples per class (min/mean/max): {train_stats['samples_per_class']['min']} / "
          f"{train_stats['samples_per_class']['mean']:.1f} / {train_stats['samples_per_class']['max']}")

    test_dataset = test_loader.dataset
    test_stats = get_dataset_statistics(test_dataset)

    print(f"\n📊 Test Set:")
    print(f"  Total samples: {test_stats['total_samples']}")
    print(f"  Number of classes: {test_stats['num_classes']}")
    print(f"  Samples per class (min/mean/max): {test_stats['samples_per_class']['min']} / "
          f"{test_stats['samples_per_class']['mean']:.1f} / {test_stats['samples_per_class']['max']}")

    print(f"\n🔢 Batch Configuration:")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    print("="*60 + "\n")
