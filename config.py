import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Output directory
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
DECISIONS_DIR = os.path.join(OUTPUTS_DIR, "decisions")

# Ensure directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, DECISIONS_DIR]:
    os.makedirs(path, exist_ok=True)

# Dataset paths (located outside the ABSA project folder, at the final project root)
RAW_TSV_PATH = os.path.join(os.path.dirname(BASE_DIR), "dataset", "amazon_reviews_us_Electronics_v1_00.tsv", "amazon_reviews_us_Electronics_v1_00.tsv")
CLEANED_PARQUET_PATH = os.path.join(PROCESSED_DATA_DIR, "reviews_clean.parquet")

# Aspect Keywords for Aspect Extraction (ABSA)
ASPECT_KEYWORDS = {
    "harga": ["expensive", "cheap", "worth", "overpriced", "affordable", "price", "cost", "value"],
    "kualitas": ["broken", "durable", "quality", "material", "build", "sturdy", "defective", "excellent"],
    "pengiriman": ["shipping", "delivery", "arrived", "package", "fast", "slow", "damaged", "late"],
    "layanan": ["customer service", "refund", "return", "support", "warranty", "replacement"]
}

# Decision Thresholds
THRESHOLD_PERTAHANKAN = 0.65
THRESHOLD_EVALUASI = 0.35

# Model file paths
MODEL_XGBOOST_PATH = os.path.join(MODELS_DIR, "quality_filter_xgboost.pkl")
MODEL_RF_PATH = os.path.join(MODELS_DIR, "quality_filter_rf.pkl")
MODEL_BASELINE_SVM_PATH = os.path.join(MODELS_DIR, "baseline_svm_absa.pkl")
