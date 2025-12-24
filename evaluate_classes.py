import torch
from collections import defaultdict
from training.dataset_loader import create_dataloaders
from models.model import create_model
from utils.config import Config


def evaluate_per_class():
    print("="*70)
    print("PER-CLASS EVALUATION")
    print("="*70)
    print()

    device = Config.get_device()
    model = create_model(
        checkpoint_path=str(Config.get_checkpoint_path()),
        device=device
    )
    model.eval()

    _, test_loader = create_dataloaders()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    print("Evaluating test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            for label, pred in zip(labels, predicted):
                label_idx = label.item()
                class_total[label_idx] += 1
                if label_idx == pred.item():
                    class_correct[label_idx] += 1

    print()
    print("="*70)
    print("RESULTS BY CLASS")
    print("="*70)
    print(f"{'Class':<50} {'Samples':<10} {'Accuracy':<10}")
    print("-"*70)

    total_correct = 0
    total_samples = 0

    for i in range(Config.NUM_CLASSES):
        if class_total[i] > 0:
            accuracy = 100.0 * class_correct[i] / class_total[i]
            class_name = Config.CLASS_NAMES_SIMPLE[i]

            if accuracy >= 60:
                marker = "✓"
            elif accuracy >= 40:
                marker = "○"
            else:
                marker = "✗"

            print(f"{marker} {class_name:<48} {class_total[i]:<10} {accuracy:>6.2f}%")

            total_correct += class_correct[i]
            total_samples += class_total[i]

    print("-"*70)
    overall_acc = 100.0 * total_correct / total_samples
    print(f"{'OVERALL':<50} {total_samples:<10} {overall_acc:>6.2f}%")
    print("="*70)
    print()

    print("CLASSES NEEDING IMPROVEMENT (Acc < 40%):")
    for i in range(Config.NUM_CLASSES):
        if class_total[i] > 0:
            accuracy = 100.0 * class_correct[i] / class_total[i]
            if accuracy < 40:
                print(f"  ⚠️  {Config.CLASS_NAMES_SIMPLE[i]}: {accuracy:.1f}% ({class_total[i]} samples)")

    print()
    print("BEST PERFORMING CLASSES (Acc >= 60%):")
    for i in range(Config.NUM_CLASSES):
        if class_total[i] > 0:
            accuracy = 100.0 * class_correct[i] / class_total[i]
            if accuracy >= 60:
                print(f"  ✓  {Config.CLASS_NAMES_SIMPLE[i]}: {accuracy:.1f}% ({class_total[i]} samples)")

    print()
    print("="*70)


if __name__ == '__main__':
    evaluate_per_class()
