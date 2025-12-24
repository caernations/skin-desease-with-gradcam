import sys
from training.dataset_loader import create_dataloaders, print_dataset_info
from training.train import train_model
from utils.config import Config

def main():
    print("="*70)
    print("TRAINING WITH CLASS BALANCING")
    print("="*70)
    print()
    print("Features enabled:")
    print("  ✓ Weighted sampling (sample minority classes more)")
    print("  ✓ Class weights in loss (penalize errors on minority classes more)")
    print("  ✓ Early stopping (auto-stop when no improvement)")
    print()
    print("="*70)
    print()

    if not Config.validate_dataset():
        print("❌ Dataset validation failed!")
        sys.exit(1)

    print("Creating balanced dataloaders...")
    train_loader, test_loader = create_dataloaders(
        use_weighted_sampling=True  
    )

    print_dataset_info(train_loader, test_loader)
    print("Starting training with class balancing...")
    model, history = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=50, 
        device=Config.get_device()
    )

    print("\n" + "="*70)
    print("✅ Training complete!")
    print(f"Best model saved: {Config.get_checkpoint_path()}")
    print("="*70)

if __name__ == '__main__':
    main()
