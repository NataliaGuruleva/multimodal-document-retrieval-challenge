from config import SEED
from utils.seed import set_seed

set_seed(SEED)

from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
from tqdm.auto import tqdm
import argparse

from retrieval.image_retriever import ImageEncoder
from retrieval.captioner import ImageCaptioner
from config import (
    IMAGE_MODEL_NAME, IMAGE_BATCH_SIZE,
    CAPTION_MODEL_NAME, CAPTION_BATCH_SIZE,
    CAPTION_MAX_NEW_TOKENS, CAPTION_NUM_BEAMS
)

from retrieval.bm25_index import BM25DocumentIndex
from retrieval.dense_retriever import DenseRetriever, DenseDocumentIndex
from retrieval.hybrid_search import HybridSearchEngine
from utils.text import tokenize, build_query_text, clean_m2kr_passage_content, normalize_text


HF_DATASET = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
DEFAULT_TASK = "EVQA"
DEFAULT_SPLIT = "test"

DATA_DIR = Path("data/M2KR-HF-local")
ARTIFACTS_DIR = Path("artifacts")

BM25_TOP_K = 200
DENSE_TOP_K = 100
FINAL_TOP_K = 5
TEXT_ALPHA = 0.6

def recall_at_k(pred: List[str], gt: Set[str], k: int) -> float:
    if not gt:
        return 0.0
    return 1.0 if any(pid in gt for pid in pred[:k]) else 0.0

def sample_queries(ds_q, n: int, seed: int = SEED):
    n = min(int(n), len(ds_q))
    ds_s = ds_q.sample(n, random_state=SEED)
    return ds_s

def build_micro_corpus_and_queries(ds_q_sample, ds_passages, corpus_size: int, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """
    Create passages corpus.
    Return:
      - df_passages: passage_id, passage_content
      - df_queries: question_id, instruction, question, img_path
      - gt_map: question_id -> set(pos_item_ids)
    """
    q_rows = []
    gt_map: Dict[str, Set[str]] = {}
    pos_passages: Dict[str, str] = {}

    for i, ex in ds_q_sample.iterrows():
        qid = str(ex.get("question_id"))
        q = normalize_text(ex.get("question"))
        inst = normalize_text(ex.get("instruction"))
        img_path = ex.get("img_path").replace("inat/", "")
        pos_ids = ex.get("pos_item_ids") or []
        pos_txts = ex.get("pos_item_contents") or []

        # gt set
        pos_set = set(str(x) for x in pos_ids if isinstance(x, (str, int)))
        gt_map[qid] = pos_set

        for pid, txt in zip(pos_ids, pos_txts):
            pid = str(pid)
            if pid not in pos_passages:
                pos_passages[pid] = clean_m2kr_passage_content(txt)
        q_rows.append({"question_id": qid, "question": q, "instruction": inst, "img_path": img_path if isinstance(img_path, str) else None})

    df_queries = pd.DataFrame(q_rows)
    corpus: Dict[str, str] = dict(pos_passages)
    target = max(int(corpus_size), len(corpus))
    
    if len(corpus) < target:
        print(f"Sampling negatives from passages to reach corpus_size={target} ...")
        ds_neg = ds_passages.sample(frac=1, random_state=seed)

        for i, ex in ds_neg.iterrows():
            pid = str(ex.get("passage_id"))
            if pid in corpus:
                continue
            txt = ex.get("passage_content") or ""
            txt = clean_m2kr_passage_content(txt)
            if not txt:
                continue
            corpus[pid] = txt
            if len(corpus) >= target:
                break

    df_passages = pd.DataFrame([{"passage_id": pid, "passage_content": txt} for pid, txt in corpus.items()])

    print(f"Micro-corpus passages: {len(df_passages)}")
    print(f"Queries: {len(df_queries)}")

    return df_passages, df_queries, gt_map


def build_or_load_bm25(task: str, split: str, df_passages: pd.DataFrame) -> BM25DocumentIndex:
    path = ARTIFACTS_DIR / f"bm25/m2kr_hf_{task.lower()}_{split}_bm25.pkl"
    if path.exists():
        print(f"Loading BM25 index from {path}")
        return BM25DocumentIndex.load(path)

    print("Building BM25 index")
    doc_ids = df_passages["passage_id"].astype(str).tolist()
    texts = df_passages["passage_content"].fillna("").astype(str).tolist()
    bm25 = BM25DocumentIndex.build(doc_ids=doc_ids, texts=texts, tokenizer=tokenize)

    path.parent.mkdir(parents=True, exist_ok=True)
    bm25.save(path)
    print(f"BM25 index saved to {path}")
    return bm25


def build_or_load_dense(task: str, split: str, df_passages: pd.DataFrame) -> Tuple[DenseRetriever, DenseDocumentIndex]:
    path = ARTIFACTS_DIR / f"embeddings/m2kr_hf_{task.lower()}_{split}_dense"
    dense_retriever = DenseRetriever()

    if path.with_suffix(".npz").exists():
        print(f"Loading Dense index from {path}")
        dense_index = DenseDocumentIndex.load(path)
        return dense_retriever, dense_index

    print("Building Dense index")
    doc_ids = df_passages["passage_id"].astype(str).tolist()
    texts = df_passages["passage_content"].fillna("").astype(str).tolist()

    dense_index = dense_retriever.build_index(doc_ids=doc_ids, texts=texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    dense_index.save(path)
    print(f"Dense index saved to {path}")
    return dense_retriever, dense_index

def evaluate_m2kr_local(engine: HybridSearchEngine, df_queries: pd.DataFrame, gt_map: Dict[str, Set[str]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    hits1 = 0.0
    hits3 = 0.0
    hits5 = 0.0
    mrr5 = 0.0
    total = 0.0

    with pred_path.open("w", encoding="utf-8") as f:
        for row in tqdm(df_queries.itertuples(index=False), total=len(df_queries), desc="Eval M2KR local"):
            qid = str(getattr(row, "question_id"))
            question = getattr(row, "question", "")
            instruction = getattr(row, "instruction", "")
            img_path = getattr(row, "img_path", "")
            query_text = build_query_text(instruction=instruction, question=question)

            pred_pids = engine.search_single(query_text=query_text, query_image_path=img_path if img_path else None, caption_prompt=None)
            gt = gt_map.get(qid, set())

            total += 1.0
            hits1 += recall_at_k(pred_pids, gt, 1)
            hits3 += recall_at_k(pred_pids, gt, 3)
            hits5 += recall_at_k(pred_pids, gt, 5)

            f.write(json.dumps({"question_id": qid, "pred": pred_pids, "gt": list(gt)}, ensure_ascii=False) + "\n")

    metrics = {
        "num_queries": int(total),
        "recall@1": float(hits1 / total) if total else 0.0,
        "recall@3": float(hits3 / total) if total else 0.0,
        "recall@5": float(hits5 / total) if total else 0.0,
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nLocal M2KR metrics (micro-corpus)")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nSaved predictions: {pred_path}")
    print(f"Saved metrics: {metrics_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default=DEFAULT_TASK)
    ap.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    ap.add_argument("--n_queries", type=int, default=500)
    ap.add_argument("--corpus_size", type=int, default=5000)

    ap.add_argument("--alpha", type=float, default=TEXT_ALPHA)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--bm25_top_k", type=int, default=BM25_TOP_K)
    ap.add_argument("--dense_top_k", type=int, default=DENSE_TOP_K)
    ap.add_argument("--final_top_k", type=int, default=FINAL_TOP_K)

    ap.add_argument("--use_image", type=int, default=0)
    ap.add_argument("--use_caption", type=int, default=0)
    ap.add_argument("--caption_append_to_query", type=int, default=1)

    args = ap.parse_args()

    ds_q = pd.read_parquet(DATA_DIR / "data/test-00000-of-00001.parquet")
    ds_p = pd.read_parquet(DATA_DIR / "passages/test_passages-00000-of-00001.parquet")
    ds_q_sample = sample_queries(ds_q, n=args.n_queries, seed=SEED)

    df_passages, df_queries, gt_map = build_micro_corpus_and_queries(
        ds_q_sample=ds_q_sample,
        ds_passages=ds_p,
        corpus_size=args.corpus_size,
        seed=SEED,
    )

    bm25 = build_or_load_bm25(task=args.task, split=args.split, df_passages=df_passages)
    dense_retriever, dense_index = build_or_load_dense(task=args.task, split=args.split, df_passages=df_passages)

    image_encoder = None
    captioner = None

    if args.use_image:
        try:
            image_encoder = ImageEncoder(model_name=IMAGE_MODEL_NAME, batch_size=IMAGE_BATCH_SIZE)
        except Exception as e:
            print(f"[WARN] ImageEncoder init failed: {e}")
            image_encoder = None

    if args.use_caption:
        try:
            captioner = ImageCaptioner(
                model_name=CAPTION_MODEL_NAME,
                batch_size=CAPTION_BATCH_SIZE,
                max_new_tokens=CAPTION_MAX_NEW_TOKENS,
                num_beams=CAPTION_NUM_BEAMS,
                show_progress=False,
            )
        except Exception as e:
            print(f"[WARN] Captioner init failed: {e}")
            captioner = None

    engine = HybridSearchEngine(
        bm25_index=bm25,
        dense_index=dense_index,
        dense_retriever=dense_retriever,
        alpha=args.alpha,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.final_top_k,
        passages_df=df_passages,

        # multimodal toggles
        image_encoder=image_encoder,
        image_alpha=args.beta if image_encoder is not None else 0.0,
        captioner=captioner,
        caption_gamma=args.gamma if captioner is not None else 0.0,
        caption_append_to_query=bool(args.caption_append_to_query),

        vlm_retriever=None,
    )

    tag = f"evqa_{args.split}_q{len(df_queries)}_c{len(df_passages)}_a{args.alpha}_b{args.beta}_g{args.gamma}_img{args.use_image}_cap{args.use_caption}"
    out_dir = ARTIFACTS_DIR / f"local_eval/{tag}"
    evaluate_m2kr_local(engine, df_queries=df_queries, gt_map=gt_map, out_dir=out_dir)


if __name__ == "__main__":
    main()