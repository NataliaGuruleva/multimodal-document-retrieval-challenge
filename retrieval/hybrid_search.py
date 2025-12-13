from pathlib import Path
from typing import Dict, List, Iterable, Optional
import json
import pandas as pd
from tqdm import tqdm
import torch

from retrieval.bm25_index import BM25DocumentIndex
from retrieval.dense_retriever import DenseRetriever, DenseDocumentIndex
from retrieval.image_retriever import ImageIndex, ImageEncoder
from utils.text import build_query_text, tokenize, normalize_text
from utils.image import load_image
from config import DATA_DIR


class HybridSearchEngine:
    """
    Hybrid search engine.
    Image branch is used ONLY if query_image_path is provided.
    """

    def __init__(
        self,
        bm25_index: BM25DocumentIndex,
        dense_index: DenseDocumentIndex,
        dense_retriever: DenseRetriever,
        alpha: float,
        bm25_top_k: int = 100,
        dense_top_k: int = 100,
        vlm_top_k: int = 20,
        final_top_k: int = 5,
        image_index: Optional[ImageIndex] = None,  # unused for M2KR
        image_encoder: Optional[ImageEncoder] = None,
        image_alpha: float = 0.3,
        vlm_retriever=None,
        passages_df=None,
    ):
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.dense_retriever = dense_retriever

        self.alpha = alpha
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k
        self.vlm_top_k = vlm_top_k

        self.image_index = image_index
        self.image_encoder = image_encoder
        self.image_alpha = image_alpha

        self.vlm_retriever = vlm_retriever
        self.passages_df = passages_df


    def _has_image_branch(self) -> bool:
        return self.image_encoder is not None


    def search_single(
        self,
        query_text: str,
        doc_name: Optional[str] = None,
        query_image_path: Optional[str] = None,
    ) -> List[str]:

        bm25_hits = self.bm25_index.retrieve(
            query=query_text,
            tokenizer=tokenize,
            top_k=self.bm25_top_k,
            doc_name=doc_name,
        )

        if not bm25_hits:
            return []

        candidate_ids = [pid for pid, _ in bm25_hits]

        query_emb = self.dense_retriever.encode_text(query_text)

        dense_hits = self.dense_index.retrieve_subset(
            query_emb=query_emb,
            candidate_ids=candidate_ids,
            top_k=self.dense_top_k,
        )


        if doc_name is not None and self.vlm_retriever is not None and self.passages_df is not None:
            ids = [pid for pid, _ in dense_hits]
            dense_top_ids = [pid for pid, _ in dense_hits[: self.vlm_top_k]]
            texts = []
            imgs = []

            for pid in dense_top_ids:
                row = self.passages_df[self.passages_df["passage_id"] == pid].iloc[0]
                text = row.get("vlm_text") or row.get("ocr_text") or ""
                img = DATA_DIR / "MMDocIR-Challenge" / row.get("image_path")
                texts.append(text)
                imgs.append(img)

            vlm_hits = self.vlm_retriever.retrieve(
                query_text=query_text,
                candidate_texts=texts,
                candidate_image_paths=imgs,
                candidate_ids=dense_top_ids,
                top_k=self.final_top_k
            )

            return [pid for pid, _ in vlm_hits]

        image_hits = None
        if query_image_path and self._has_image_branch():
            try:
                query_img_emb = self.image_encoder.encode_images(["M2KR-Challenge/passage_images/" + query_image_path])
                id_to_idx = {pid: i for i, pid in enumerate(self.dense_index.doc_ids)}
                passage_texts = [normalize_text(self.passages_df[self.passages_df["passage_id"] == pid].iloc[0].get("passage_content") or "") for pid in candidate_ids]
                passage_text_embs = self.image_encoder.encode_texts(passage_texts)
                scores = torch.matmul(query_img_emb, passage_text_embs.T).squeeze(0)
                image_hits = list(zip(candidate_ids, scores.tolist()))
            except Exception as e:
                image_hits = None

        final_hits = DenseRetriever.hybrid_top_k(
            bm25_scores=bm25_hits,
            dense_scores=dense_hits,
            image_scores=image_hits,
            alpha=self.alpha,
            beta=self.image_alpha if image_hits is not None else 0.0,
            top_k=self.final_top_k,
        )

        return [pid for pid, _ in final_hits]


def run_m2kr_hybrid(df_queries: pd.DataFrame, engine: HybridSearchEngine, output_path: Path) -> None:
    """
    Run hybrid retrieval for M2KR.

    df_queries columns:
        - question_id
        - instruction
        - question
        - img_path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for _, row in tqdm(df_queries.iterrows(), total=len(df_queries), desc="M2KR hybrid search"):
            qid = str(row["question_id"])
            query_text = build_query_text(instruction=row.get("instruction"), question=row.get("question"))
            img_path = row["img_path"] if isinstance(row.get("img_path"), str) else None

            passage_ids = engine.search_single(query_text=query_text, doc_name=None, query_image_path=img_path)

            f.write(json.dumps({"question_id": qid, "passage_id": passage_ids}) + "\n")


def run_mmdoc_hybrid(queries: Iterable[Dict], engine: HybridSearchEngine, output_path: Path) -> None:
    """Run hybrid retrieval for MMDocIR."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for item in tqdm(queries, desc="MMDocIR hybrid search"):
            qid = str(item["question_id"])
            query_text = item["question"]
            doc_name = item["doc_name"]

            passage_ids = engine.search_single(query_text=query_text, doc_name=doc_name, query_image_path=None)

            f.write(json.dumps({"question_id": qid, "passage_id": passage_ids}) + "\n")


def run_hybrid_pipeline(
    mode: str,
    bm25_index: BM25DocumentIndex,
    dense_index: DenseDocumentIndex,
    dense_retriever: DenseRetriever,
    queries,
    output_path: str | Path,
    alpha: float,
    bm25_top_k: int,
    dense_top_k: int,
    final_top_k: int,
    image_index: Optional[ImageIndex] = None,
    image_encoder: Optional[ImageEncoder] = None,
    image_alpha: float = 0.3,
    vlm_retriever=None,          # ← добавлено
    passages_df=None,            # ← добавлено
):
    """Unified pipeline entry point."""

    engine = HybridSearchEngine(
        bm25_index=bm25_index,
        dense_index=dense_index,
        dense_retriever=dense_retriever,
        alpha=alpha,
        bm25_top_k=bm25_top_k,
        dense_top_k=dense_top_k,
        final_top_k=final_top_k,
        image_index=image_index,
        image_encoder=image_encoder,
        image_alpha=image_alpha,
        vlm_retriever=vlm_retriever, # for future updates
        passages_df=passages_df,
    )

    output_path = Path(output_path)

    if mode == "m2kr":
        run_m2kr_hybrid(df_queries=queries, engine=engine, output_path=output_path)
    elif mode == "mmdoc":
        run_mmdoc_hybrid(queries=queries, engine=engine, output_path=output_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
