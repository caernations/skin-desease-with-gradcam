import streamlit as st
from utils.config import Config


def setup_page_config():
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout=Config.LAYOUT,
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/skin-disease-classifier',
            'Report a bug': 'https://github.com/yourusername/skin-disease-classifier/issues',
            'About': """
            # Skin Disease Classifier

            An AI-powered tool for skin disease classification with Grad-CAM interpretability.

            Built with PyTorch and Streamlit.
            """
        }
    )


def apply_custom_css():
    st.markdown("""
    <style>
        /* Main title styling */
        .main-title {
            text-align: center;
            color: #1f77b4;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }

        /* Subtitle styling */
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }

        /* Card styling */
        .card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            margin: 1rem 0;
        }

        /* Prediction result styling */
        .prediction-box {
            padding: 2rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            margin: 1rem 0;
        }

        /* Confidence badge */
        .confidence-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            background-color: rgba(255, 255, 255, 0.2);
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 1rem;
        }

        /* Info box */
        .info-box {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.25rem;
        }

        /* Warning box */
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.25rem;
        }

        /* Success box */
        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.25rem;
        }

        /* Image container */
        .image-container {
            border: 2px solid #ddd;
            border-radius: 0.5rem;
            padding: 0.5rem;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Metric card */
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #666;
            padding: 2rem 0;
            margin-top: 3rem;
            border-top: 1px solid #ddd;
        }

        /* Streamlit default element adjustments */
        .stButton>button {
            width: 100%;
            background-color: #1f77b4;
            color: white;
            border-radius: 0.5rem;
            padding: 0.75rem;
            font-weight: bold;
            border: none;
        }

        .stButton>button:hover {
            background-color: #155a8a;
        }

        /* File uploader styling */
        .uploadedFile {
            border: 2px dashed #1f77b4;
            border-radius: 0.5rem;
        }

        /* Hide Streamlit branding (optional) */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
    </style>
    """, unsafe_allow_html=True)


def show_header():
    st.markdown(
        f'<h1 class="main-title">{Config.PAGE_ICON} Skin Disease Classifier</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">AI-Powered Dermatology Assistant with Grad-CAM Interpretability</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def show_footer():
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <p>
                Built with ❤️ using PyTorch, EfficientNet, and Streamlit<br>
                <strong>Disclaimer:</strong> This tool is for educational purposes only.
                Not a substitute for professional medical advice.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_two_column_layout():
    return st.columns(2)


def create_three_column_layout():
    return st.columns(3)


def create_tabs(tab_names: list):
    return st.tabs(tab_names)


def show_divider():
    st.markdown("---")


def show_spacer(height: int = 2):
    for _ in range(height):
        st.write("")


def create_centered_container():
    col1, col2, col3 = st.columns([1, 3, 1])
    return col2


def show_success_message(message: str):
    st.success(f"✅ {message}")


def show_error_message(message: str):
    st.error(f"❌ {message}")


def show_warning_message(message: str):
    st.warning(f"⚠️ {message}")


def show_info_message(message: str):
    st.info(f"ℹ️ {message}")
