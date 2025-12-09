from pathlib import Path
import json
import pandas as pd
from retrieval.bm25_index import BM25DocumentIndex
from retrieval.dense_retriever import DenseRetriever, DenseDocumentIndex
from retrieval.image_retriever import ImageEncoder, ImageIndex
from retrieval.hybrid_search import run_hybrid_pipeline
from utils.text import normalize_text, tokenize


# Parameters
DATA_DIR = Path("data/MMDocIR-Challenge")
ARTIFACTS_DIR = Path("artifacts")
PASSAGE_FILE = DATA_DIR / "MMDocIR_pages.parquet"
QUERY_FILE = DATA_DIR / "MMDocIR_gt_remove.jsonl"
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25/mmdoc_bm25.pkl"
DENSE_INDEX_PATH = ARTIFACTS_DIR / "embeddings/mmdoc_dense"
IMAGE_INDEX_PATH = ARTIFACTS_DIR / "embeddings/mmdoc_image.pt"
OUTPUT_PATH = ARTIFACTS_DIR / "submission/submission_mmdoc.jsonl"

# Retrieval parameters
BM25_TOP_K = 200
DENSE_TOP_K = 100
FINAL_TOP_K = 5
TEXT_ALPHA = 0.6 # dense-text weight
IMAGE_ALPHA = 0.1 # image


def load_mmdoc_data():
    print("Loading MMDocIR passages")
    df_passages = pd.read_parquet(PASSAGE_FILE)

    print("Loading MMDocIR queries")
    queries = []
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line.strip()))

    return df_passages, queries


def build_bm25_index(df_passages: pd.DataFrame) -> BM25DocumentIndex:
    """Build BM25 index with doc_name restriction support."""
    print("Building BM25 index")

    doc_ids = df_passages["passage_id"].astype(str).tolist()
    texts = []
    doc_names = []

    for _, row in df_passages.iterrows():
        text = row.get("vlm_text") or row.get("ocr_text") or ""
        texts.append(normalize_text(text))
        doc_names.append(row["doc_name"])

    bm25 = BM25DocumentIndex.build(
        doc_ids=doc_ids,
        texts=texts,
        tokenizer=tokenize,
        doc_names=doc_names,
    )

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    bm25.save(BM25_INDEX_PATH)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")

    return bm25


def build_dense_index(df_passages: pd.DataFrame):
    print("Initializing DenseRetriever")
    dense_retriever = DenseRetriever()

    if DENSE_INDEX_PATH.with_suffix(".npz").exists():
        print(f"Loading Dense index from {DENSE_INDEX_PATH}")
        dense_index = DenseDocumentIndex.load(DENSE_INDEX_PATH)
        return dense_retriever, dense_index

    print("Building Dense index")
    doc_ids = df_passages["passage_id"].astype(str).tolist()
    texts = [
        normalize_text(row.get("vlm_text") or row.get("ocr_text") or "")
        for _, row in df_passages.iterrows()
    ]

    dense_index = dense_retriever.build_index(doc_ids=doc_ids, texts=texts)
    dense_index.save(DENSE_INDEX_PATH)

    print(f"Dense index saved to {DENSE_INDEX_PATH}")
    return dense_retriever, dense_index


def build_image_index_if_available(df_passages: pd.DataFrame):
    """
    Not used by default
    """
    if "image_path" not in df_passages.columns:
        return None, None

    if IMAGE_INDEX_PATH.exists():
        print(f"Loading Image index from {IMAGE_INDEX_PATH}")
        image_index = ImageIndex.load(IMAGE_INDEX_PATH)
        image_encoder = ImageEncoder()
        return image_encoder, image_index

    print("Building Image index (MMDocIR)")

    id_to_path = {}
    for _, row in df_passages.iterrows():
        pid = str(row["passage_id"])
        path = row.get("image_path")
        if isinstance(path, str) and Path(path).exists():
            id_to_path[pid] = path
        else:
            id_to_path[pid] = None

    image_encoder = ImageEncoder()
    image_index = ImageIndex.build_from_paths(image_encoder, id_to_path)

    IMAGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    image_index.save(IMAGE_INDEX_PATH)

    print(f"Image index saved to {IMAGE_INDEX_PATH}")
    return image_encoder, image_index


def main():
    df_passages, queries = load_mmdoc_data()
    bm25_index = build_bm25_index(df_passages)
    dense_retriever, dense_index = build_dense_index(df_passages)

    # image fusion disabled by default
    image_encoder, image_index = None, None

    print("Running hybrid retrieval")

    run_hybrid_pipeline(
        mode="mmdoc",
        bm25_index=bm25_index,
        dense_index=dense_index,
        dense_retriever=dense_retriever,
        queries=queries,
        output_path=OUTPUT_PATH,
        alpha=TEXT_ALPHA,
        bm25_top_k=BM25_TOP_K,
        dense_top_k=DENSE_TOP_K,
        final_top_k=FINAL_TOP_K,
        image_index=image_index,
        image_encoder=image_encoder,
        image_alpha=IMAGE_ALPHA,
    )

    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()