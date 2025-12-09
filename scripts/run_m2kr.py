from pathlib import Path
import pandas as pd
from retrieval.bm25_index import BM25DocumentIndex
from retrieval.dense_retriever import DenseRetriever, DenseDocumentIndex
from retrieval.image_retriever import ImageEncoder, ImageIndex
from retrieval.hybrid_search import run_hybrid_pipeline
from utils.text import normalize_text, tokenize


DATA_DIR = Path("data/M2KR-Challenge")
ARTIFACTS_DIR = Path("artifacts")
PASSAGE_FILE = DATA_DIR / "challenge_passage/train-00000-of-00001.parquet"
QUERY_FILE = DATA_DIR / "challenge_data/train-00000-of-00001.parquet"
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25/m2kr_bm25.pkl"
DENSE_INDEX_PATH = ARTIFACTS_DIR / "embeddings/m2kr_dense"
OUTPUT_PATH = ARTIFACTS_DIR / "submission/submission_m2kr.jsonl"
BM25_TOP_K = 200
DENSE_TOP_K = 100
FINAL_TOP_K = 5

TEXT_ALPHA = 0.6     # dense-text weight (BM25 + dense)
IMAGE_ALPHA = 0.3    # image weight


def load_m2kr_data():
    print("Loading M2KR passages")
    df_passages = pd.read_parquet(PASSAGE_FILE)

    print("Loading M2KR queries")
    df_queries = pd.read_parquet(QUERY_FILE)
    if "id" in df_queries.columns and "question_id" not in df_queries.columns:
        df_queries = df_queries.rename(columns={"id": "question_id"})

    return df_passages, df_queries


def build_or_load_bm25(df_passages: pd.DataFrame) -> BM25DocumentIndex:
    if BM25_INDEX_PATH.exists():
        print(f"Loading BM25 index from {BM25_INDEX_PATH}")
        return BM25DocumentIndex.load(BM25_INDEX_PATH)

    print("Building BM25 index")

    doc_ids = df_passages["passage_id"].astype(str).tolist()
    texts = (df_passages["passage_content"].fillna("").map(normalize_text).tolist())

    bm25 = BM25DocumentIndex.build(doc_ids=doc_ids, texts=texts, tokenizer=tokenize)
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    bm25.save(BM25_INDEX_PATH)

    print(f"BM25 index saved to {BM25_INDEX_PATH}")
    return bm25


def build_or_load_dense(df_passages: pd.DataFrame):
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


def build_query_image_encoder() -> ImageEncoder:
    """In M2KR images are available ONLY at query level."""
    print("Initializing ImageEncoder for query images")
    return ImageEncoder()


def main():
    df_passages, df_queries = load_m2kr_data()
    bm25_index = build_or_load_bm25(df_passages)
    dense_retriever, dense_index = build_or_load_dense(df_passages)
    # for future
    image_encoder = build_query_image_encoder()
    image_index = None

    print("Running hybrid retrieval")

    run_hybrid_pipeline(
        mode="m2kr",
        bm25_index=bm25_index,
        dense_index=dense_index,
        dense_retriever=dense_retriever,
        queries=df_queries,
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