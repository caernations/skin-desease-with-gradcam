import argparse
import sys
from pathlib import Path

from utils.config import Config


def train_model(args):
    print("Starting model training...\n")
    from training.dataset_loader import create_dataloaders, print_dataset_info
    from training.train import train_model
    if not Config.validate_dataset():
        print("❌ Dataset validation failed. Please check your dataset structure.")
        return

    train_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampling=args.weighted_sampling
    )

    if args.verbose:
        print_dataset_info(train_loader, test_loader)

    # Train
    model, history = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        device=args.device,
        resume_from=args.resume
    )

    print("\n✅ Training completed successfully!")
    print(f"   Best model saved at: {Config.get_checkpoint_path()}")


def run_inference(args):
    print(f"Running inference on: {args.image}\n")

    from utils.inference import quick_predict
    from PIL import Image

    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return

    results = quick_predict(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\n🎯 Predicted Disease: {results['predicted_label_simple']}")
    print(f"   Confidence: {results['confidence']*100:.2f}%")
    print(f"\n📋 Full Name: {results['predicted_label']}")

    print(f"\n📊 Top 5 Predictions:")
    for i, (idx, full_name, simple_name, prob) in enumerate(results['top_k_predictions'], 1):
        print(f"   {i}. {simple_name:<40} {prob*100:>6.2f}%")

    print("="*60 + "\n")

    if args.save_overlay:
        overlay_path = Path(args.image).stem + "_gradcam.png"
        results['gradcam_overlay'].save(overlay_path)
        print(f"✓ Grad-CAM overlay saved: {overlay_path}")


def run_webapp(args):
    import subprocess
    print("Launching Streamlit web application...\n")

    cmd = [
        "streamlit", "run",
        "webapp/app.py",
        "--server.port", str(args.port),
        "--server.address", args.host
    ]

    if args.theme:
        cmd.extend(["--theme.base", args.theme])

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down web application...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch web application: {e}")


def validate_setup(args):
    print("Validating project setup...\n")

    print("1. Checking dataset...")
    if Config.validate_dataset():
        print("   ✓ Dataset found and valid")
    else:
        print("   ❌ Dataset validation failed")

    print("\n2. Checking model checkpoint...")
    checkpoint_path = Config.get_checkpoint_path()
    if checkpoint_path.exists():
        print(f"   ✓ Checkpoint found: {checkpoint_path}")
    else:
        print(f"   ⚠️  No checkpoint found at: {checkpoint_path}")
        print(f"      Run training first: python main.py train")

    print("\n3. Checking dependencies...")
    required_modules = ['torch', 'torchvision', 'streamlit', 'PIL', 'cv2', 'matplotlib']

    all_good = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✓ {module}")
        except ImportError:
            print(f"   ❌ {module} not found")
            all_good = False

    print("\n4. Checking compute device...")
    device = Config.get_device()
    print(f"   Device: {device.upper()}")

    if device == 'cuda':
        import torch
        print(f"   ✓ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
    elif device == 'mps':
        print(f"   ✓ MPS available - using Apple Silicon GPU")
    else:
        print(f"   ⚠️  Using CPU (training will be slower)")

    print("\n" + "="*60)
    if all_good and Config.validate_dataset():
        print("✅ All checks passed! Project is ready.")
    else:
        print("⚠️  Some issues detected. Please resolve them before proceeding.")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Skin Disease Classification with Grad-CAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train --epochs 50 --batch-size 32

  # Run inference on an image
  python main.py infer --image path/to/image.jpg

  # Launch web application
  python main.py webapp --port 8501

  # Validate setup
  python main.py validate
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                             help=f'Number of epochs (default: {Config.NUM_EPOCHS})')
    train_parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                             help=f'Batch size (default: {Config.BATCH_SIZE})')
    train_parser.add_argument('--num-workers', type=int, default=Config.NUM_WORKERS,
                             help=f'Number of workers (default: {Config.NUM_WORKERS})')
    train_parser.add_argument('--device', type=str, default=None,
                             help='Device to use (cuda/mps/cpu)')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Resume training from checkpoint')
    train_parser.add_argument('--weighted-sampling', action='store_true',
                             help='Use weighted sampling for class imbalance')
    train_parser.add_argument('--verbose', action='store_true',
                             help='Print detailed information')

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on an image')
    infer_parser.add_argument('--image', type=str, required=True,
                             help='Path to input image')
    infer_parser.add_argument('--checkpoint', type=str, default=None,
                             help='Path to model checkpoint')
    infer_parser.add_argument('--device', type=str, default=None,
                             help='Device to use (cuda/mps/cpu)')
    infer_parser.add_argument('--save-overlay', action='store_true',
                             help='Save Grad-CAM overlay image')

    # Webapp command
    webapp_parser = subparsers.add_parser('webapp', help='Launch Streamlit web app')
    webapp_parser.add_argument('--port', type=int, default=8501,
                               help='Port to run on (default: 8501)')
    webapp_parser.add_argument('--host', type=str, default='localhost',
                               help='Host to bind to (default: localhost)')
    webapp_parser.add_argument('--theme', type=str, choices=['light', 'dark'],
                               help='UI theme')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate project setup')

    args = parser.parse_args()
    if args.command == 'train':
        train_model(args)
    elif args.command == 'infer':
        run_inference(args)
    elif args.command == 'webapp':
        run_webapp(args)
    elif args.command == 'validate':
        validate_setup(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
