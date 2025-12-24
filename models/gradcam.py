import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                        device: str = 'cpu') -> np.ndarray:
        self.model.eval()
        self.model.to(device)
        input_tensor = input_tensor.to(device)

        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        gradients = self.gradients[0].cpu().numpy()  # Shape: (C, H, W)
        activations = self.activations[0].cpu().numpy()  # Shape: (C, H, W)
        weights = np.mean(gradients, axis=(1, 2))  # Shape: (C,)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # Shape: (H, W)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def overlay_heatmap(self, heatmap: np.ndarray, original_image: Image.Image,
                       alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> Image.Image:
        img_width, img_height = original_image.size
        heatmap_resized = cv2.resize(heatmap, (img_width, img_height))

        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        original_np = np.array(original_image)

        if original_np.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(
                heatmap_colored,
                (original_np.shape[1], original_np.shape[0])
            )

        overlayed = cv2.addWeighted(
            original_np,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )

        # Convert back to PIL Image
        return Image.fromarray(overlayed)

    def generate_visualization(self, input_tensor: torch.Tensor, original_image: Image.Image,
                              target_class: Optional[int] = None, device: str = 'cpu',
                              alpha: float = 0.4) -> Tuple[np.ndarray, Image.Image]:
        heatmap = self.generate_heatmap(input_tensor, target_class, device)
        overlayed = self.overlay_heatmap(heatmap, original_image, alpha)
        return heatmap, overlayed

    def __del__(self):
        pass


def create_gradcam(model: nn.Module, target_layer: Optional[nn.Module] = None) -> GradCAM:
    if target_layer is None:
        if hasattr(model, 'get_last_conv_layer'):
            target_layer = model.get_last_conv_layer()
        else:
            raise ValueError("Please provide target_layer or implement get_last_conv_layer() method")

    return GradCAM(model, target_layer)
