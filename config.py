from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

BM25_DIR = ARTIFACTS_DIR / "bm25"
EMB_DIR = ARTIFACTS_DIR / "embeddings"
SUBMISSION_DIR = ARTIFACTS_DIR / "submission"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BM25_K1 = 1.5
BM25_B = 0.75
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_BATCH_SIZE = 64
DENSE_POOLING = "mean" # future extensions
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_BATCH_SIZE = 16
# Text fusion: BM25 + Dense
HYBRID_ALPHA = 0.6       # weight of dense-text

# Multimodal fusion: (text-hybrid) + image
HYBRID_BETA = 0.3        # weight of image branch
BM25_TOP_K = 200
DENSE_TOP_K = 100
FINAL_TOP_K = 5

# Captioning (M2KR query images)
CAPTION_MODEL_NAME = "C:/Users/dimaa/blip_model"
CAPTION_BATCH_SIZE = 8
CAPTION_MAX_NEW_TOKENS = 32
CAPTION_NUM_BEAMS = 3

# If True, caption text is appended to query_text for BM25 + dense
CAPTION_APPEND_TO_QUERY = True

# Text fusion: BM25 + Dense + CaptionDense + Image
HYBRID_GAMMA = 0.0       # weight of caption-dense branch
CAPTION_TOP_K = 100

SEED = 42
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
