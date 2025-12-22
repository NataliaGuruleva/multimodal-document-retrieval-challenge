from config import SEED
from utils.seed import set_seed
import argparse

set_seed(SEED)

from pathlib import Path
import json
import pandas as pd

from retrieval.bm25_index import BM25DocumentIndex
from retrieval.dense_retriever import DenseRetriever, DenseDocumentIndex
from retrieval.image_retriever import ImageEncoder, ImageIndex
from retrieval.hybrid_search import run_hybrid_pipeline
from retrieval.vlm_retriever import VLMRetriever
from utils.text import normalize_text, tokenize
from utils.image import resolve_image_path


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
TEXT_ALPHA = 0.0
IMAGE_ALPHA = 0.0


def load_mmdoc_data(passage_file: Path, query_file: Path):
    print("Loading MMDocIR passages")
    df_passages = pd.read_parquet(passage_file)

    print("Loading MMDocIR queries")
    queries = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line.strip()))

    return df_passages, queries


def build_bm25_index(df_passages: pd.DataFrame, out_path: Path) -> BM25DocumentIndex:
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bm25.save(out_path)

    print(f"BM25 index saved to {out_path}")
    return bm25


def build_dense_index(df_passages: pd.DataFrame, out_path: Path):
    print("Initializing DenseRetriever")
    dense_retriever = DenseRetriever()

    if out_path.with_suffix(".npz").exists():
        print(f"Loading Dense index from {out_path}")
        dense_index = DenseDocumentIndex.load(out_path)
        return dense_retriever, dense_index

    print("Building Dense index")

    doc_ids = df_passages["passage_id"].astype(str).tolist()
    texts = [
        normalize_text(row.get("vlm_text") or row.get("ocr_text") or "")
        for _, row in df_passages.iterrows()
    ]

    dense_index = dense_retriever.build_index(doc_ids=doc_ids, texts=texts)
    dense_index.save(out_path)

    print(f"Dense index saved to {out_path}")
    return dense_retriever, dense_index


def build_image_index_if_available(df_passages: pd.DataFrame):
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
        rp = resolve_image_path(path) if isinstance(path, str) else None
        id_to_path[pid] = str(rp) if rp is not None else None

    image_encoder = ImageEncoder()
    image_index = ImageIndex.build_from_paths(image_encoder, id_to_path)

    IMAGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    image_index.save(IMAGE_INDEX_PATH)

    print(f"Image index saved to {IMAGE_INDEX_PATH}")
    return image_encoder, image_index


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

    ap.add_argument("--alpha", type=float, default=TEXT_ALPHA)
    ap.add_argument("--beta", type=float, default=IMAGE_ALPHA)

    ap.add_argument("--use_vlm", type=int, default=1)
    ap.add_argument("--vlm_model_name", type=str, default="C:/Users/dimaa/siglip_model")
    args = ap.parse_args()

    df_passages, queries = load_mmdoc_data(
        passage_file=Path(args.passage_file),
        query_file=Path(args.query_file),
    )

    bm25_path = Path(args.bm25_index_path)
    if bm25_path.exists():
        print(f"Loading BM25 index from {bm25_path}")
        bm25_index = BM25DocumentIndex.load(bm25_path)
    else:
        bm25_index = build_bm25_index(df_passages, out_path=bm25_path)

    dense_path = Path(args.dense_index_path)
    dense_retriever, dense_index = build_dense_index(df_passages, out_path=dense_path)

    vlm = None
    if args.use_vlm:
        try:
            vlm = VLMRetriever(model_name=args.vlm_model_name)
        except Exception as e:
            print(f"[WARN] VLM init failed: {e}")
            vlm = None

    # optional: image branch
    image_encoder, image_index = None, None

    print("Running hybrid retrieval")

    run_hybrid_pipeline(
        mode="mmdoc",
        bm25_index=bm25_index,
        dense_index=dense_index,
        dense_retriever=dense_retriever,
        queries=queries,
        output_path=args.output_path,
        alpha=args.alpha,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.final_top_k,
        image_index=image_index,
        image_encoder=image_encoder,
        image_alpha=args.beta if vlm is not None else 0.0,
        vlm_retriever=vlm,
        passages_df=df_passages,
        captioner=None,
        caption_gamma=0.0
    )

    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
