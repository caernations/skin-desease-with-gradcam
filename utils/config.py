import os
from pathlib import Path


class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATASET_DIR = PROJECT_ROOT / "dataset"
    TRAIN_DIR = DATASET_DIR / "train"
    TEST_DIR = DATASET_DIR / "test"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Model settings
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 19
    DROPOUT_RATE = 0.3

    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 10

    # Image settings
    IMG_SIZE = 224  
    IMG_MEAN = [0.485, 0.456, 0.406]  
    IMG_STD = [0.229, 0.224, 0.225]

    # Data augmentation
    ROTATION_DEGREES = 20
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.2
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2
    COLOR_JITTER_SATURATION = 0.2

    # Training settings
    NUM_WORKERS = 4
    PIN_MEMORY = False  
    SHUFFLE_TRAIN = True

    # Class names (19 skin diseases)
    CLASS_NAMES = [
        "Acne and Rosacea Photos",
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
        "Atopic Dermatitis Photos",
        "Cellulitis Impetigo and other Bacterial Infections",
        "Eczema Photos",
        "Exanthems and Drug Eruptions",
        "Herpes HPV and other STDs Photos",
        "Light Diseases and Disorders of Pigmentation",
        "Lupus and other Connective Tissue diseases",
        "Melanoma Skin Cancer Nevi and Moles",
        "Poison Ivy Photos and other Contact Dermatitis",
        "Psoriasis pictures Lichen Planus and related diseases",
        "Seborrheic Keratoses and other Benign Tumors",
        "Systemic Disease",
        "Tinea Ringworm Candidiasis and other Fungal Infections",
        "Urticaria Hives",
        "Vascular Tumors",
        "Vasculitis Photos",
        "Warts Molluscum and other Viral Infections"
    ]

    # Simplified class names for display
    CLASS_NAMES_SIMPLE = [
        "Acne & Rosacea",
        "Skin Cancer (Malignant)",
        "Atopic Dermatitis",
        "Bacterial Infections",
        "Eczema",
        "Drug Eruptions",
        "Viral STDs",
        "Pigmentation Disorders",
        "Lupus & Tissue Diseases",
        "Melanoma & Moles",
        "Contact Dermatitis",
        "Psoriasis",
        "Benign Tumors",
        "Systemic Disease",
        "Fungal Infections",
        "Urticaria (Hives)",
        "Vascular Tumors",
        "Vasculitis",
        "Viral Warts"
    ]

    # Device settings
    @staticmethod
    def get_device():
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    # Grad-CAM settings
    GRADCAM_ALPHA = 0.4  
    GRADCAM_COLORMAP = "jet"  

    # Streamlit UI settings
    PAGE_TITLE = "Skin Disease Classifier"
    PAGE_ICON = "🔬"
    LAYOUT = "wide"

    # Checkpoint settings
    CHECKPOINT_FILENAME = "best_model.pth"

    @classmethod
    def create_directories(cls):
        cls.CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)
        cls.LOGS_DIR.mkdir(exist_ok=True, parents=True)

    @classmethod
    def get_checkpoint_path(cls, filename: str = None) -> Path:
        if filename is None:
            filename = cls.CHECKPOINT_FILENAME
        return cls.CHECKPOINTS_DIR / filename

    @classmethod
    def validate_dataset(cls) -> bool:
        if not cls.TRAIN_DIR.exists():
            print(f"❌ Training directory not found: {cls.TRAIN_DIR}")
            return False
        if not cls.TEST_DIR.exists():
            print(f"❌ Test directory not found: {cls.TEST_DIR}")
            return False
        print(f"✓ Dataset validated")
        return True


Config.create_directories()
