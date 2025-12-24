import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import time
from pathlib import Path
import json

from models.model import SkinDiseaseModel, create_model
from utils.config import Config


class Trainer:
    def __init__(self,
                 model: SkinDiseaseModel,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 device: Optional[str] = None,
                 save_dir: Optional[Path] = None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device if device else Config.get_device()
        self.save_dir = save_dir if save_dir else Config.CHECKPOINTS_DIR

        self.model.to(self.device)
        class_weights = self._calculate_class_weights()

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = criterion

        self.optimizer = optimizer if optimizer else optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        print(f"✓ Trainer initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Optimizer: {self.optimizer.__class__.__name__}")
        print(f"  - Criterion: {self.criterion.__class__.__name__}")
        print(f"  - Using class weights for imbalanced dataset")

    def _calculate_class_weights(self) -> torch.Tensor:
        class_counts = torch.zeros(Config.NUM_CLASSES)

        for _, labels in self.train_loader:
            for label in labels:
                class_counts[label] += 1

        total_samples = class_counts.sum()
        class_weights = total_samples / (Config.NUM_CLASSES * class_counts)
        class_weights = torch.clamp(class_weights, max=10.0)

        print(f"  - Class weight range: {class_weights.min():.2f} to {class_weights.max():.2f}")

        return class_weights

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if (batch_idx + 1) % 10 == 0:
                print(f'  Batch [{batch_idx + 1}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100. * correct / total:.2f}%')

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / len(self.test_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs: Optional[int] = None,
             early_stopping_patience: Optional[int] = None,
             save_best: bool = True) -> Dict:

        num_epochs = num_epochs if num_epochs else Config.NUM_EPOCHS
        early_stopping_patience = early_stopping_patience if early_stopping_patience else Config.EARLY_STOPPING_PATIENCE

        print("\n" + "="*60)
        print(f"Starting Training for {num_epochs} epochs")
        print("="*60 + "\n")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1

            print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            print("-" * 40)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            print(f"\n{'='*40}")
            print(f"Epoch {self.current_epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0

                if save_best:
                    self.save_checkpoint(
                        filename='best_model.pth',
                        is_best=True
                    )
                    print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")

            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{early_stopping_patience}")

            print(f"{'='*40}\n")

            if self.patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {self.current_epoch} epochs")
                print(f"   Best Val Loss: {self.best_val_loss:.4f}")
                print(f"   Best Val Acc: {self.best_val_acc:.2f}%")
                break

            if self.current_epoch % 5 == 0:
                self.save_checkpoint(filename=f'checkpoint_epoch_{self.current_epoch}.pth')

        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        print("="*60 + "\n")

        self.save_checkpoint(filename='final_model.pth')
        self.save_history()

        return self.history

    def save_checkpoint(self, filename: str, is_best: bool = False):
        checkpoint_path = self.save_dir / filename

        metrics = {
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else None,
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
        }

        self.model.save_checkpoint(
            filepath=str(checkpoint_path),
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict(),
            metrics=metrics
        )

    def save_history(self, filename: str = 'training_history.json'):
        history_path = self.save_dir / filename

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"✓ Training history saved: {history_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = self.model.load_checkpoint(checkpoint_path, device=self.device)
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            self.best_val_loss = metrics.get('best_val_loss', float('inf'))
            self.best_val_acc = metrics.get('best_val_acc', 0.0)

        print(f"✓ Checkpoint loaded - Ready to resume from epoch {self.current_epoch}")


def train_model(train_loader: DataLoader,
               test_loader: DataLoader,
               num_epochs: Optional[int] = None,
               device: Optional[str] = None,
               resume_from: Optional[str] = None) -> Tuple[SkinDiseaseModel, Dict]:

    model = create_model(
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        device=device
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )

    if resume_from:
        trainer.load_checkpoint(resume_from)

    history = trainer.train(num_epochs=num_epochs)
    return model, history


if __name__ == '__main__':
    print("Initializing training pipeline...\n")

    if not Config.validate_dataset():
        print("❌ Dataset validation failed. Please check your dataset structure.")
        exit(1)

    from training.dataset_loader import create_dataloaders, print_dataset_info

    train_loader, test_loader = create_dataloaders(
        use_weighted_sampling=False  
    )

    print_dataset_info(train_loader, test_loader)
    model, history = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=Config.NUM_EPOCHS
    )

    print("\n✅ Training completed successfully!")
    print(f"   Best model saved at: {Config.get_checkpoint_path()}")
