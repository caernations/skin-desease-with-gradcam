import streamlit as st
from PIL import Image
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from webapp.layout import (
    setup_page_config,
    apply_custom_css,
    show_header,
    show_footer,
    show_divider,
    show_success_message,
    show_error_message,
    show_warning_message,
    create_tabs
)
from webapp.components import (
    image_uploader,
    display_prediction_results,
    display_top_predictions,
    display_gradcam_visualization,
    sidebar_info,
    show_upload_instructions,
    show_model_details,
    display_metrics_dashboard,
    create_download_button,
    loading_spinner
)
from utils.inference import InferenceEngine
from utils.config import Config


@st.cache_resource
def load_inference_engine():
    try:
        checkpoint_path = str(Config.get_checkpoint_path())
        engine = InferenceEngine(checkpoint_path=checkpoint_path)
        return engine, None
    except Exception as e:
        return None, str(e)


def main():
    setup_page_config()
    apply_custom_css()
    show_header()
    sidebar_info()
    with st.spinner("Loading AI model..."):
        engine, error = load_inference_engine()

    if error:
        show_error_message(f"Failed to load model: {error}")
        st.info("""
        **Model not found!**

        Please train the model first by running:
        ```bash
        uv run python training/train.py
        ```

        Or if you have a pre-trained model, place it in the `checkpoints/` directory
        with the name `best_model.pth`.
        """)
        return

    with st.expander("ℹ️ Model Information"):
        model_info = engine.get_model_info()
        display_metrics_dashboard(model_info)

    show_divider()

    tab1, tab2, tab3 = create_tabs(["🔍 Classify Image", "📊 Model Details", "ℹ️ How to Use"])
    with tab1:
        st.markdown("## Upload and Classify")
        show_upload_instructions()

        uploaded_image = image_uploader()

        if uploaded_image:
            st.markdown("### Uploaded Image")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(uploaded_image, caption="Uploaded Image", width='stretch')

            show_divider()

            if st.button("🔬 Analyze Image", type="primary"):
                with loading_spinner("Analyzing image... This may take a few seconds..."):
                    try:
                        results = engine.predict_with_explanation(
                            uploaded_image,
                            top_k=5,
                            alpha=Config.GRADCAM_ALPHA
                        )

                        show_success_message("Analysis complete!")
                        show_divider()
                        display_prediction_results(results)
                        show_divider()
                        display_top_predictions(results, top_k=5)
                        show_divider()
                        display_gradcam_visualization(
                            original_image=uploaded_image,
                            overlayed_image=results['gradcam_overlay'],
                            heatmap=results['gradcam_heatmap']
                        )

                        show_divider()

                        st.markdown("### 💾 Download Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            create_download_button(
                                results['gradcam_overlay'],
                                filename="gradcam_overlay.png"
                            )

                        with col2:
                            st.info("Download the Grad-CAM overlay to share or save the results.")

                        show_divider()
                        show_warning_message(
                            "This is an AI prediction and should not replace professional medical diagnosis. "
                            "Please consult a dermatologist for accurate diagnosis and treatment."
                        )

                    except Exception as e:
                        show_error_message(f"Error during analysis: {str(e)}")
                        st.exception(e)
        else:
            st.info("👆 Please upload an image to begin analysis.")

    with tab2:
        show_model_details()

        st.markdown("### 📋 Supported Skin Conditions")
        col1, col2 = st.columns(2)

        mid_point = len(Config.CLASS_NAMES_SIMPLE) // 2

        with col1:
            for i, name in enumerate(Config.CLASS_NAMES_SIMPLE[:mid_point], 1):
                st.markdown(f"**{i}.** {name}")

        with col2:
            for i, name in enumerate(Config.CLASS_NAMES_SIMPLE[mid_point:], mid_point + 1):
                st.markdown(f"**{i}.** {name}")

        st.markdown("---")

        st.markdown("### 🏗️ Training Configuration")
        st.json({
            "Model": Config.MODEL_NAME,
            "Image Size": f"{Config.IMG_SIZE}x{Config.IMG_SIZE}",
            "Batch Size": Config.BATCH_SIZE,
            "Learning Rate": Config.LEARNING_RATE,
            "Epochs": Config.NUM_EPOCHS,
            "Early Stopping Patience": Config.EARLY_STOPPING_PATIENCE
        })

    with tab3:
        st.markdown("""
        ## 🎯 How to Use This Application

        ### Step 1: Upload an Image
        - Click on the "Browse files" button in the "Classify Image" tab
        - Select a clear image of a skin lesion
        - Supported formats: JPG, JPEG, PNG

        ### Step 2: Analyze
        - Click the "Analyze Image" button
        - Wait for the AI to process the image (usually takes a few seconds)

        ### Step 3: Review Results
        The application will show you:
        - **Main Prediction**: The most likely skin condition
        - **Confidence Score**: How confident the model is (0-100%)
        - **Top 5 Predictions**: Alternative diagnoses ranked by probability
        - **Grad-CAM Visualization**: Heatmap showing which parts of the image influenced the prediction

        ### Understanding Grad-CAM
        - **Red/Yellow areas**: Regions the model focused on most
        - **Blue/Purple areas**: Regions with less influence
        - This helps verify if the model is looking at the right features
        """)

    show_footer()


if __name__ == "__main__":
    main()
