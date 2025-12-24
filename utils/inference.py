import torch
from PIL import Image
from typing import Dict, Tuple, Optional
import numpy as np
from models.model import SkinDiseaseModel, create_model
from models.gradcam import GradCAM, create_gradcam
from .preprocessing import preprocess_image
from .config import Config


class InferenceEngine:
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device if device else Config.get_device()
        self.checkpoint_path = checkpoint_path if checkpoint_path else str(Config.get_checkpoint_path())
        self.model = self._load_model()
        self.gradcam = create_gradcam(self.model)

        print(f"✓ InferenceEngine initialized on device: {self.device}")

    def _load_model(self) -> SkinDiseaseModel:
        model = create_model(
            num_classes=Config.NUM_CLASSES,
            pretrained=True,
            checkpoint_path=self.checkpoint_path,
            device=self.device
        )
        model.eval()
        return model

    def predict(self, image: Image.Image, top_k: int = 5) -> Dict:
        input_tensor = preprocess_image(image, mode='inference')
        predicted_classes, probabilities = self.model.predict(input_tensor, device=self.device)
        predicted_class = predicted_classes[0].item()
        confidence = probabilities[0, predicted_class].item()
        top_k_probs, top_k_indices = torch.topk(probabilities[0], k=min(top_k, Config.NUM_CLASSES))

        top_k_predictions = [
            (
                idx.item(),
                Config.CLASS_NAMES[idx.item()],
                Config.CLASS_NAMES_SIMPLE[idx.item()],
                prob.item()
            )
            for idx, prob in zip(top_k_indices, top_k_probs)
        ]

        return {
            'predicted_class': predicted_class,
            'predicted_label': Config.CLASS_NAMES[predicted_class],
            'predicted_label_simple': Config.CLASS_NAMES_SIMPLE[predicted_class],
            'confidence': confidence,
            'top_k_predictions': top_k_predictions,
            'all_probabilities': probabilities[0].cpu().numpy()
        }

    def generate_gradcam(self, image: Image.Image, target_class: Optional[int] = None,
                        alpha: float = None) -> Tuple[np.ndarray, Image.Image]:
        if alpha is None:
            alpha = Config.GRADCAM_ALPHA

        input_tensor = preprocess_image(image, mode='inference')
        heatmap, overlayed = self.gradcam.generate_visualization(
            input_tensor=input_tensor,
            original_image=image,
            target_class=target_class,
            device=self.device,
            alpha=alpha
        )

        return heatmap, overlayed

    def predict_with_explanation(self, image: Image.Image, top_k: int = 5,
                                 alpha: float = None) -> Dict:

        prediction_results = self.predict(image, top_k=top_k)
        heatmap, overlayed = self.generate_gradcam(
            image,
            target_class=prediction_results['predicted_class'],
            alpha=alpha
        )

        results = {
            **prediction_results,
            'gradcam_heatmap': heatmap,
            'gradcam_overlay': overlayed
        }

        return results

    def predict_batch(self, images: list[Image.Image], top_k: int = 5) -> list[Dict]:
        results = []

        for image in images:
            result = self.predict(image, top_k=top_k)
            results.append(result)

        return results

    def reload_model(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()
        self.gradcam = create_gradcam(self.model)
        print(f"✓ Model reloaded from: {checkpoint_path}")

    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = self.model.get_trainable_params()

        return {
            'model_name': Config.MODEL_NAME,
            'num_classes': Config.NUM_CLASSES,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': self.device,
            'checkpoint_path': self.checkpoint_path
        }


def quick_predict(image_path: str, checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None) -> Dict:
    image = Image.open(image_path).convert('RGB')
    engine = InferenceEngine(checkpoint_path=checkpoint_path, device=device)
    results = engine.predict_with_explanation(image)

    return results
