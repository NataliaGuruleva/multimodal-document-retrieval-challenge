from config import SEED
from utils.seed import set_seed

set_seed(SEED)

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json
import pandas as pd
from datasets import load_dataset

from retrieval.bm25_index import BM25DocumentIndex
from retrieval.dense_retriever import DenseRetriever, DenseDocumentIndex
from retrieval.vlm_retriever import VLMRetriever
from retrieval.hybrid_search import HybridSearchEngine
from utils.text import normalize_text, tokenize


DATASET_NAME = "vidore/docvqa_test_subsampled_ocr_chunk"
SPLIT = "test"

DATA_DIR = Path("data/ViDoRe-DocVQA")
IMG_DIR = DATA_DIR / "page_images"
CACHE_PAGES_PARQUET = DATA_DIR / "pages.parquet"
CACHE_QUERIES_JSONL = DATA_DIR / "queries.jsonl"

ARTIFACTS_DIR = Path("artifacts")
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25/vidore_docvqa_bm25.pkl"
DENSE_INDEX_PATH = ARTIFACTS_DIR / "embeddings/vidore_docvqa_dense"
OUTPUT_PRED_JSONL = ARTIFACTS_DIR / "submission/vidore_docvqa_pred.jsonl"

BM25_TOP_K = 200
DENSE_TOP_K = 100
FINAL_TOP_K = 5
TEXT_ALPHA = 0.3
IMAGE_ALPHA = 0.1
MAX_PAGE_CHARS = 6000


def _page_pid(doc_id: str, page: str) -> str:
    return f"{doc_id}_{page}"


def prepare_pages_and_queries() -> Tuple[pd.DataFrame, List[Dict], Dict[str, str]]:
    """
    Create page-level dataset from chunk-level.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # cache hit
    if CACHE_PAGES_PARQUET.exists() and CACHE_QUERIES_JSONL.exists():
        df_pages = pd.read_parquet(CACHE_PAGES_PARQUET)
        df_pages["doc_name"] = "1"
        queries = []
        gt = {}
        with CACHE_QUERIES_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                queries.append({"question_id": obj["question_id"], "question": obj["question"], "doc_name": "1"})
                gt[obj["question_id"]] = obj["gt_passage_id"]
        return df_pages, queries, gt

    print(f"Loading dataset: {DATASET_NAME} [{SPLIT}]")
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    # page accumulator
    page_texts: Dict[str, List[str]] = defaultdict(list)
    page_doc: Dict[str, str] = {}
    page_img_path: Dict[str, str] = {}

    # queries + gt
    queries_dict: Dict[str, Dict] = {}
    gt: Dict[str, str] = {}

    n = len(ds)
    print(f"Rows: {n}")

    for i, ex in enumerate(ds):
        doc_id = str(ex["docId"])
        page = str(ex["page"])
        pid = _page_pid(doc_id, page)
        page_doc[pid] = doc_id
        txt = normalize_text(ex.get("text_description"))
        ctype = ex.get("chunk_type")
        if txt:
            if isinstance(ctype, str) and ctype.strip():
                page_texts[pid].append(f"[{ctype}] {txt}")
            else:
                page_texts[pid].append(txt)
        if pid not in page_img_path:
            img = ex.get("image", None)
            if img is not None:
                out_path = IMG_DIR / f"{pid}.png"
                try:
                    img.convert("RGB").save(out_path)
                    page_img_path[pid] = str(out_path)
                except Exception:
                    page_img_path[pid] = ""
        qid = str(ex["questionId"])
        if qid not in queries_dict:
            qtext = normalize_text(ex.get("query"))
            queries_dict[qid] = {"question_id": qid, "question": qtext, "doc_name": "1"}
            gt[qid] = pid
        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{n}")

    rows = []
    for pid, doc_id in page_doc.items():
        text = " ".join(page_texts.get(pid, []))
        text = normalize_text(text)
        if MAX_PAGE_CHARS and len(text) > MAX_PAGE_CHARS:
            text = text[:MAX_PAGE_CHARS]
        rows.append({"passage_id": pid, "doc_name": "1", "ocr_text": text, "image_path": page_img_path.get(pid, "")})

    df_pages = pd.DataFrame(rows)
    print("Pages:", len(df_pages), "Queries:", len(queries_dict))

    df_pages.to_parquet(CACHE_PAGES_PARQUET, index=False)
    with CACHE_QUERIES_JSONL.open("w", encoding="utf-8") as f:
        for q in queries_dict.values():
            obj = dict(q)
            obj["gt_passage_id"] = gt[q["question_id"]]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return df_pages, list(queries_dict.values()), gt

def build_or_load_bm25(df_pages: pd.DataFrame) -> BM25DocumentIndex:
    if BM25_INDEX_PATH.exists():
        print(f"Loading BM25 index from {BM25_INDEX_PATH}")
        return BM25DocumentIndex.load(BM25_INDEX_PATH)

    print("Building BM25 index")
    doc_ids = df_pages["passage_id"].astype(str).tolist()
    doc_names = df_pages["doc_name"].astype(str).tolist()
    texts = df_pages["ocr_text"].fillna("").astype(str).tolist()

    bm25 = BM25DocumentIndex.build(doc_ids=doc_ids, texts=texts, tokenizer=tokenize, doc_names=doc_names)
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    bm25.save(BM25_INDEX_PATH)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")
    return bm25

def build_or_load_dense(df_pages: pd.DataFrame) -> Tuple[DenseRetriever, DenseDocumentIndex]:
    dense_retriever = DenseRetriever()

    if DENSE_INDEX_PATH.with_suffix(".npz").exists():
        print(f"Loading Dense index from {DENSE_INDEX_PATH}")
        dense_index = DenseDocumentIndex.load(DENSE_INDEX_PATH)
        return dense_retriever, dense_index

    print("Building Dense index")
    doc_ids = df_pages["passage_id"].astype(str).tolist()
    texts = df_pages["ocr_text"].fillna("").astype(str).tolist()

    dense_index = dense_retriever.build_index(doc_ids=doc_ids, texts=texts)
    dense_index.save(DENSE_INDEX_PATH)
    print(f"Dense index saved to {DENSE_INDEX_PATH}")
    return dense_retriever, dense_index

def evaluate(engine: HybridSearchEngine, queries: List[Dict], gt: Dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hits1 = 0
    hits3 = 0
    hits5 = 0
    total = 0

    with out_path.open("w", encoding="utf-8") as f:
        for q in queries:
            qid = str(q["question_id"])
            qtext = q["question"]
            doc_name = q["doc_name"]
            gt_pid = gt.get(qid)
            pred_pids = engine.search_single(query_text=qtext, doc_name=q["doc_name"], query_image_path=None)
            f.write(json.dumps({"question_id": qid, "passage_id": pred_pids}, ensure_ascii=False) + "\n")
            total += 1
            if gt_pid:
                if len(pred_pids) > 0 and pred_pids[0] == gt_pid:
                    hits1 += 1
                if gt_pid in pred_pids[:3]:
                    hits3 += 1
                if gt_pid in pred_pids[:5]:
                    hits5 += 1

    if total == 0:
        print("No queries to evaluate.")
        return

    print(f"Eval queries: {total}")
    print(f"Recall@1: {hits1 / total:.4f}")
    print(f"Recall@3: {hits3 / total:.4f}")
    print(f"Recall@5: {hits5 / total:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=TEXT_ALPHA)
    ap.add_argument("--bm25_top_k", type=int, default=BM25_TOP_K)
    ap.add_argument("--dense_top_k", type=int, default=DENSE_TOP_K)
    ap.add_argument("--final_top_k", type=int, default=FINAL_TOP_K)
    ap.add_argument("--use_vlm", type=int, default=1)
    ap.add_argument("--image_alpha", type=float, default=IMAGE_ALPHA)
    args = ap.parse_args()

    df_pages, queries, gt = prepare_pages_and_queries()

    bm25_index = build_or_load_bm25(df_pages)
    dense_retriever, dense_index = build_or_load_dense(df_pages)

    vlm = None
    if args.use_vlm:
        try:
            vlm = VLMRetriever()
        except Exception as e:
            print(f"[WARN] VLM init failed: {e}")
            vlm = None

    engine = HybridSearchEngine(
        bm25_index=bm25_index,
        dense_index=dense_index,
        dense_retriever=dense_retriever,
        alpha=args.alpha,
        image_alpha=args.image_alpha,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.final_top_k,
        vlm_retriever=vlm,
        passages_df=df_pages,
    )

    tag = f"vidore_{SPLIT}_a{args.alpha}_vlm{args.use_vlm}"
    out_path = ARTIFACTS_DIR / f"local_eval/{tag}.jsonl"

    print("Running Evaluation")
    evaluate(engine, queries, gt, out_path)
    print(f"Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()