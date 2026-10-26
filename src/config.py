"""Shared paths and constants for the RNN SMS spam detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "SMSSpamCollection.tsv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"

TOKENIZER_FILE = MODELS_DIR / "tokenizer.json"
BEST_MODEL_FILE = MODELS_DIR / "best_model.keras"
BEST_HISTORY_FILE = MODELS_DIR / "best_history.json"
BEST_CONFIG_FILE = MODELS_DIR / "best_config.json"
RESULTS_FILE = OUTPUTS_DIR / "experiments_results.json"
TEST_METRICS_FILE = OUTPUTS_DIR / "test_metrics.json"

SEED = 42
VOCAB_SIZE = 5000      # top-N words kept (index 0 = padding, 1 = OOV)
MAX_LEN = 50           # tokens per message after padding/truncation
VAL_SIZE = 0.15
TEST_SIZE = 0.15
THRESHOLD = 0.5        # sigmoid output >= threshold -> spam
