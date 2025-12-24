import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import io

from .config import Config


def plot_predictions(probabilities: np.ndarray, class_names: List[str] = None,
                    top_k: int = 5, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    if class_names is None:
        class_names = Config.CLASS_NAMES_SIMPLE

    top_k = min(top_k, len(probabilities))
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_probs = probabilities[top_indices]
    top_labels = [class_names[i] for i in top_indices]
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn(top_probs)  
    y_pos = np.arange(len(top_labels))

    bars = ax.barh(y_pos, top_probs, color=colors, alpha=0.8, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels, fontsize=10)
    ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Predictions', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])

    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.2%}',
                ha='left', va='center', fontsize=10, fontweight='bold')

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_gradcam(original_image: Image.Image, heatmap: np.ndarray,
                overlayed_image: Image.Image,
                predicted_label: str, confidence: float,
                figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet', alpha=0.8)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(overlayed_image)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Add prediction info
    fig.suptitle(f'Prediction: {predicted_label} (Confidence: {confidence:.2%})',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def create_results_figure(original_image: Image.Image,
                          prediction_results: dict,
                          heatmap: np.ndarray,
                          overlayed_image: Image.Image,
                          top_k: int = 5,
                          figsize: Tuple[int, int] = (18, 10)) -> plt.Figure:

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Top row: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Original image
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Heatmap
    im = ax2.imshow(heatmap, cmap='jet', alpha=0.8)
    ax2.set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Overlay
    ax3.imshow(overlayed_image)
    ax3.set_title('Interpretability Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Bottom row: Predictions
    ax4 = fig.add_subplot(gs[1, :])

    # Plot predictions bar chart
    probabilities = prediction_results['all_probabilities']
    top_k_actual = min(top_k, len(probabilities))
    top_indices = np.argsort(probabilities)[-top_k_actual:][::-1]
    top_probs = probabilities[top_indices]
    top_labels = [Config.CLASS_NAMES_SIMPLE[i] for i in top_indices]

    colors = plt.cm.RdYlGn(top_probs)
    y_pos = np.arange(len(top_labels))

    bars = ax4.barh(y_pos, top_probs, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_labels, fontsize=11)
    ax4.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax4.set_title(f'Top {top_k_actual} Predictions', fontsize=13, fontweight='bold')
    ax4.set_xlim([0, 1])

    # Add probability values
    for bar, prob in zip(bars, top_probs):
        width = bar.get_width()
        ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.2%}',
                ha='left', va='center', fontsize=10, fontweight='bold')

    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)

    predicted_label = prediction_results['predicted_label_simple']
    confidence = prediction_results['confidence']
    fig.suptitle(f'Skin Disease Classification Results\nPredicted: {predicted_label} ({confidence:.2%} confidence)',
                fontsize=16, fontweight='bold', y=0.98)

    return fig


def save_figure_to_bytes(fig: plt.Figure, format: str = 'png', dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def fig_to_pil(fig: plt.Figure, dpi: int = 150) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img


def create_comparison_grid(images: List[Image.Image], titles: List[str] = None,
                          rows: int = 2, cols: int = 3,
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=11, fontweight='bold')
        ax.axis('off')

    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig
