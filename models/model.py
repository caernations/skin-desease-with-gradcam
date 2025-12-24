import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Optional
import os


class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes: int = 19, pretrained: bool = True, dropout: float = 0.3):
        super(SkinDiseaseModel, self).__init__()
        self.num_classes = num_classes
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_last_conv_layer(self) -> nn.Module:
        return self.backbone.features[-1]

    def predict(self, x: torch.Tensor, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        self.to(device)
        x = x.to(device)

        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

        return predicted_classes, probabilities

    def save_checkpoint(self, filepath: str, epoch: int = 0, optimizer_state: Optional[dict] = None,
                       metrics: Optional[dict] = None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'optimizer_state_dict': optimizer_state,
            'metrics': metrics or {}
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str, device: str = 'cpu', load_optimizer: bool = False) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(device)

        print(f"Checkpoint loaded: {filepath}")
        print(f"- Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"- Classes: {checkpoint.get('num_classes', 'N/A')}")

        return checkpoint

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes: int = 19, pretrained: bool = True,
                checkpoint_path: Optional[str] = None, device: str = 'cpu') -> SkinDiseaseModel:
    model = SkinDiseaseModel(num_classes=num_classes, pretrained=pretrained)

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_checkpoint(checkpoint_path, device=device)

    model.to(device)
    return model
