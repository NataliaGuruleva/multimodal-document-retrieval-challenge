from config import SEED
from utils.seed import set_seed

set_seed(SEED)

import argparse
from pathlib import Path
import pandas as pd
from retrieval.bm25_index import BM25DocumentIndex
from retrieval.dense_retriever import DenseRetriever, DenseDocumentIndex
from retrieval.image_retriever import ImageEncoder, ImageIndex
from retrieval.hybrid_search import run_hybrid_pipeline
from utils.text import normalize_text, tokenize
from utils.text import clean_m2kr_passage_content
from retrieval.captioner import ImageCaptioner
from config import (
    CAPTION_MODEL_NAME,
    CAPTION_BATCH_SIZE,
    CAPTION_MAX_NEW_TOKENS,
    CAPTION_NUM_BEAMS,
    CAPTION_APPEND_TO_QUERY,
    HYBRID_GAMMA,
    CAPTION_TOP_K,
    IMAGE_MODEL_NAME,
    IMAGE_BATCH_SIZE,
)

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

TEXT_ALPHA = 0.5     # dense-text weight
IMAGE_ALPHA = 0.3    # image weight


def load_m2kr_data(passage_file: Path, query_file: Path, strip_inat_prefix: bool = False):
    print("Loading M2KR passages")
    df_passages = pd.read_parquet(passage_file)

    print("Loading M2KR queries")
    df_queries = pd.read_parquet(query_file)
    if "id" in df_queries.columns and "question_id" not in df_queries.columns:
        df_queries = df_queries.rename(columns={"id": "question_id"})
    if strip_inat_prefix and "img_path" in df_queries.columns:
        df_queries["img_path"] = df_queries["img_path"].apply(
            lambda x: x.replace("inat/", "") if isinstance(x, str) else x
        )
    return df_passages, df_queries


def build_or_load_bm25(df_passages: pd.DataFrame) -> BM25DocumentIndex:
    if BM25_INDEX_PATH.exists():
        print(f"Loading BM25 index from {BM25_INDEX_PATH}")
        return BM25DocumentIndex.load(BM25_INDEX_PATH)

    print("Building BM25 index")

    doc_ids = df_passages["passage_id"].astype(str).tolist()
    texts = df_passages["passage_content"].fillna("").map(clean_m2kr_passage_content).tolist()

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
    texts = df_passages["passage_content"].fillna("").map(clean_m2kr_passage_content).tolist()
    dense_index = dense_retriever.build_index(doc_ids=doc_ids, texts=texts)
    dense_index.save(DENSE_INDEX_PATH)
    print(f"Dense index saved to {DENSE_INDEX_PATH}")
    return dense_retriever, dense_index


def build_query_image_encoder(model_name: str = IMAGE_MODEL_NAME, batch_size: int = IMAGE_BATCH_SIZE) -> ImageEncoder:
    """In M2KR images are available ONLY at query level."""
    print("Initializing ImageEncoder for query images")
    return ImageEncoder(model_name=model_name, batch_size=batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passage_file", type=str, default=str(PASSAGE_FILE))
    ap.add_argument("--query_file", type=str, default=str(QUERY_FILE))
    ap.add_argument("--output_path", type=str, default=str(OUTPUT_PATH))
    ap.add_argument("--bm25_index_path", type=str, default=str(BM25_INDEX_PATH))
    ap.add_argument("--dense_index_path", type=str, default=str(DENSE_INDEX_PATH))

    ap.add_argument("--bm25_top_k", type=int, default=BM25_TOP_K)
    ap.add_argument("--dense_top_k", type=int, default=DENSE_TOP_K)
    ap.add_argument("--final_top_k", type=int, default=FINAL_TOP_K)

    ap.add_argument("--alpha", type=float, default=TEXT_ALPHA)   # dense-text
    ap.add_argument("--beta", type=float, default=IMAGE_ALPHA)   # image branch
    ap.add_argument("--gamma", type=float, default=HYBRID_GAMMA) # caption branch

    ap.add_argument("--use_image", type=int, default=1)          # 0/1
    ap.add_argument("--use_caption", type=int, default=1)        # 0/1
    ap.add_argument("--caption_append_to_query", type=int, default=int(CAPTION_APPEND_TO_QUERY))  # 0/1

    ap.add_argument("--image_model_name", type=str, default=IMAGE_MODEL_NAME)
    ap.add_argument("--image_batch_size", type=int, default=IMAGE_BATCH_SIZE)

    ap.add_argument("--caption_model_name", type=str, default=CAPTION_MODEL_NAME)
    ap.add_argument("--caption_batch_size", type=int, default=CAPTION_BATCH_SIZE)
    ap.add_argument("--caption_max_new_tokens", type=int, default=CAPTION_MAX_NEW_TOKENS)
    ap.add_argument("--caption_num_beams", type=int, default=CAPTION_NUM_BEAMS)
    ap.add_argument("--caption_top_k", type=int, default=CAPTION_TOP_K)

    ap.add_argument("--strip_inat_prefix", type=int, default=0)  # 0/1
    args = ap.parse_args()

    df_passages, df_queries = load_m2kr_data(
        passage_file=Path(args.passage_file),
        query_file=Path(args.query_file),
        strip_inat_prefix=bool(args.strip_inat_prefix),
    )
    bm25_index = build_or_load_bm25(df_passages)
    dense_retriever, dense_index = build_or_load_dense(df_passages)
    image_encoder = None
    image_index = None
    if args.use_image:
        try:
            image_encoder = build_query_image_encoder(args.image_model_name, args.image_batch_size)
        except Exception as e:
            print(f"[WARN] ImageEncoder init failed: {e}")
            image_encoder = None

    captioner = None
    if args.use_caption:
        try:
            captioner = ImageCaptioner(
                model_name=args.caption_model_name,
                batch_size=args.caption_batch_size,
                max_new_tokens=args.caption_max_new_tokens,
                num_beams=args.caption_num_beams,
                show_progress=False,
            )
        except Exception as e:
            print(f"[WARN] Captioner init failed: {e}")
            captioner = None

    print("Running hybrid retrieval")

    run_hybrid_pipeline(
        mode="m2kr",
        bm25_index=bm25_index,
        dense_index=dense_index,
        dense_retriever=dense_retriever,
        queries=df_queries,
        output_path=args.output_path,
        alpha=args.alpha,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.final_top_k,
        image_index=image_index,
        image_encoder=image_encoder,
        image_alpha=args.beta if image_encoder is not None else 0.0,
        passages_df=df_passages,
        captioner=captioner,
        caption_gamma=args.gamma if captioner is not None else 0.0,
        caption_top_k=args.caption_top_k,
        caption_append_to_query=bool(args.caption_append_to_query),
    )

    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()