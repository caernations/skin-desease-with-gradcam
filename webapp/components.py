import streamlit as st
from PIL import Image
import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt

from utils.config import Config


def image_uploader(key: str = "image_upload") -> Optional[Image.Image]:
    uploaded_file = st.file_uploader(
        "Choose a skin lesion image",
        type=['jpg', 'jpeg', 'png'],
        key=key,
        help="Upload a clear image of the skin lesion for analysis"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        return image

    return None


def display_prediction_results(results: Dict):
    st.markdown("### 🎯 Prediction Result")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Predicted Disease:**")
        st.markdown(f"# {results['predicted_label_simple']}")

    with col2:
        confidence = results['confidence']
        confidence_pct = confidence * 100

        if confidence >= 0.7:
            color = "green"
        elif confidence >= 0.5:
            color = "orange"
        else:
            color = "red"

        st.markdown(f"**Confidence:**")
        st.markdown(f"<h1 style='color: {color};'>{confidence_pct:.1f}%</h1>",
                   unsafe_allow_html=True)

    with st.expander("📋 Full Classification Name"):
        st.write(results['predicted_label'])


def display_top_predictions(results: Dict, top_k: int = 5):

    st.markdown(f"### 📊 Top {top_k} Predictions")

    top_predictions = results['top_k_predictions'][:top_k]

    for i, (class_idx, full_name, simple_name, prob) in enumerate(top_predictions, 1):
        col1, col2, col3 = st.columns([0.5, 3, 1.5])

        with col1:
            if i == 1:
                st.markdown("🥇")
            elif i == 2:
                st.markdown("🥈")
            elif i == 3:
                st.markdown("🥉")
            else:
                st.markdown(f"**{i}.**")

        with col2:
            st.markdown(f"**{simple_name}**")

        with col3:
            st.progress(prob, text=f"{prob*100:.1f}%")


def display_gradcam_visualization(original_image: Image.Image,
                                 overlayed_image: Image.Image,
                                 heatmap: np.ndarray):

    st.markdown("### 🔥 Grad-CAM Interpretability")

    st.markdown("""
    **What is Grad-CAM?**
    Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the regions
    in the image that most influenced the model's prediction. Warmer colors (red/yellow)
    indicate areas the model focused on.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Image**")
        st.image(original_image, width='stretch')

    with col2:
        st.markdown("**Heatmap**")
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(heatmap, cmap='jet')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close()

    with col3:
        st.markdown("**Overlay**")
        st.image(overlayed_image, width='stretch')


def sidebar_info():
    with st.sidebar:
        st.markdown("## About")
        st.markdown("""
        This application uses a deep learning model based on **EfficientNetB0**
        to classify skin diseases from images.

        The model is trained on 19 different skin conditions and uses
        **Grad-CAM** for visual explanations.
        """)

        st.markdown("---")

        st.markdown("## 📋 Supported Conditions")
        st.markdown("The model can detect:")

        with st.expander("View all 19 conditions"):
            for i, name in enumerate(Config.CLASS_NAMES_SIMPLE, 1):
                st.markdown(f"{i}. {name}")

        st.markdown("---")

        st.markdown("## ⚙️ Model Information")
        st.markdown(f"""
        - **Architecture:** {Config.MODEL_NAME.upper()}
        - **Input Size:** {Config.IMG_SIZE}x{Config.IMG_SIZE}
        - **Classes:** {Config.NUM_CLASSES}
        """)

        st.markdown("---")

        st.markdown("## ⚠️ Disclaimer")
        st.warning("""
        **Important:** This tool is for educational purposes only.
        It should NOT be used as a substitute for professional medical diagnosis.
        Always consult a healthcare professional for medical concerns.
        """)


def show_upload_instructions():
    st.info("""
    **📸 Upload Instructions:**
    - Upload a clear, well-lit image of the skin lesion
    - Supported formats: JPG, JPEG, PNG
    - Ensure the lesion is the main focus of the image
    - Avoid blurry or low-quality images
    """)


def show_model_details():
    with st.expander("🔍 View Model Architecture Details"):
        st.markdown("""
        ### Model Architecture

        **Backbone:** EfficientNetB0
        - Pre-trained on ImageNet
        - Fine-tuned on skin disease dataset
        - Custom classification head with dropout

        **Training Details:**
        - Data augmentation: rotation, flips, color jitter
        - Optimizer: AdamW
        - Loss function: Cross-Entropy
        - Early stopping with patience

        **Grad-CAM:**
        - Visual explanation technique
        - Highlights discriminative regions
        - Uses gradients from last convolutional layer
        """)


def display_metrics_dashboard(model_info: Dict):
    st.markdown("### 📈 Model Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Parameters",
            value=f"{model_info['total_parameters']:,}"
        )

    with col2:
        st.metric(
            label="Trainable Params",
            value=f"{model_info['trainable_parameters']:,}"
        )

    with col3:
        st.metric(
            label="Classes",
            value=model_info['num_classes']
        )

    with col4:
        st.metric(
            label="Device",
            value=model_info['device'].upper()
        )


def create_download_button(image: Image.Image, filename: str = "result.png"):
    import io

    buf = io.BytesIO()
    image.save(buf, format='PNG')
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Result",
        data=byte_im,
        file_name=filename,
        mime="image/png"
    )


def show_example_images():
    st.markdown("### 📚 Example Images")

    st.info("""
    Don't have a test image? Try using sample images from the dataset
    located in `dataset/test/` directory.
    """)


def loading_spinner(message: str = "Processing..."):
    return st.spinner(message)
