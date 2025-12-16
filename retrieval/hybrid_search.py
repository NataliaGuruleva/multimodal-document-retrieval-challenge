from pathlib import Path
from typing import Dict, List, Iterable, Optional
import json
import pandas as pd
from tqdm import tqdm
import torch

from retrieval.bm25_index import BM25DocumentIndex
from retrieval.dense_retriever import DenseRetriever, DenseDocumentIndex
from retrieval.image_retriever import ImageIndex, ImageEncoder
from retrieval.captioner import ImageCaptioner
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
        vlm_top_k: int = 100,
        final_top_k: int = 5,
        image_index: Optional[ImageIndex] = None,  # unused for M2KR
        image_encoder: Optional[ImageEncoder] = None,
        image_alpha: float = 0.3,
        vlm_retriever=None,
        passages_df=None,
        captioner: Optional[ImageCaptioner] = None,
        caption_gamma: float = 0.2,
        caption_top_k: int = 100,
        caption_append_to_query: bool = True,
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
        self.captioner = captioner
        self.caption_gamma = caption_gamma
        self.caption_top_k = caption_top_k
        self.caption_append_to_query = caption_append_to_query

        self.pid2text = {}
        if self.passages_df is not None and "passage_id" in self.passages_df.columns:
            # M2KR: passage_content, MMDocIR: vlm_text/ocr_text
            if "passage_content" in self.passages_df.columns:
                for row in self.passages_df.itertuples(index=False):
                    pid = str(getattr(row, "passage_id"))
                    txt = getattr(row, "passage_content", "") or ""
                    self.pid2text[pid] = normalize_text(txt)
            else:
                for row in self.passages_df.itertuples(index=False):
                    pid = str(getattr(row, "passage_id"))
                    txt = (getattr(row, "vlm_text", None) or getattr(row, "ocr_text", None) or "")
                    self.pid2text[pid] = normalize_text(txt)


    def _has_image_branch(self) -> bool:
        return self.image_encoder is not None


    def search_single(
        self,
        query_text: str,
        doc_name: Optional[str] = None,
        query_image_path: Optional[str] = None,
        caption_prompt: Optional[str] = None
    ) -> List[str]:
        caption_text = ""
        resolved_query_img = None

        if query_image_path and self.captioner is not None:
            try:
                resolved_query_img = "M2KR-Challenge/passage_images/" + query_image_path
                caption_text = self.captioner.caption_image(resolved_query_img, prompt=caption_prompt)
            except Exception:
                caption_text = ""
                resolved_query_img = None

        query_text_for_text = query_text
        if self.caption_append_to_query and caption_text:
            query_text_for_text = f"{query_text} {caption_text}"

        bm25_hits = self.bm25_index.retrieve(
            query=query_text_for_text,
            tokenizer=tokenize,
            top_k=self.bm25_top_k,
            doc_name=doc_name,
        )

        if not bm25_hits:
            return []

        candidate_ids = [pid for pid, _ in bm25_hits]
        candidate_set = set(candidate_ids)
        query_emb = self.dense_retriever.encode_text(query_text_for_text)

        if doc_name is None:
            dense_hits = self.dense_index.retrieve(query_emb, top_k=self.dense_top_k)
            candidate_set.update([pid for pid, _ in dense_hits])
        else:
            dense_hits = self.dense_index.retrieve_subset(
                query_emb=query_emb,
                candidate_ids=candidate_ids,
                top_k=self.dense_top_k,
            )
        caption_hits = None
        if (not self.caption_append_to_query) and caption_text and self.caption_gamma > 0.0:
            try:
                cap_emb = self.dense_retriever.encode_text(caption_text)

                if doc_name is None:
                    caption_hits = self.dense_index.retrieve(cap_emb, top_k=self.caption_top_k)
                    candidate_set.update([pid for pid, _ in caption_hits])
                else:
                    caption_hits = self.dense_index.retrieve_subset(
                        query_emb=cap_emb,
                        candidate_ids=candidate_ids,
                        top_k=self.caption_top_k,
                    )
            except Exception:
                caption_hits = None

        if doc_name is not None and self.vlm_retriever is not None and self.passages_df is not None:
            doc_df = self.passages_df[self.passages_df["doc_name"] == doc_name]
            if len(doc_df) > 0:
                doc_pids = doc_df["passage_id"].astype(str).tolist()
            doc_imgs = []
            for p in doc_df.get("image_path", []).tolist():
                if isinstance(p, str) and p.strip():
                    doc_imgs.append("data/MMDocIR-Challenge/" + p)
                else:
                    doc_imgs.append(None)

            vlm_hits = self.vlm_retriever.retrieve(
                query_text=query_text,
                image_paths=doc_imgs,
                doc_page_ids=doc_pids,
                top_k=self.vlm_top_k
            )
        
        candidate_ids = sorted(candidate_set)
        image_hits = None
        if query_image_path and self._has_image_branch():
            try:
                query_img_emb = self.image_encoder.encode_images(["M2KR-Challenge/passage_images/" + query_image_path])
                id_to_idx = {pid: i for i, pid in enumerate(self.dense_index.doc_ids)}
                passage_texts = [self.pid2text.get(pid, "Empty.") for pid in candidate_ids]
                passage_text_embs = self.image_encoder.encode_texts(passage_texts)
                scores = torch.matmul(query_img_emb, passage_text_embs.T).squeeze(0)
                image_hits = list(zip(candidate_ids, scores.tolist()))
            except Exception as e:
                image_hits = None
        
        final_image_scores = image_hits
        final_beta = self.image_alpha if image_hits is not None else 0.0
        if doc_name is not None and vlm_hits is not None and len(vlm_hits) > 0:
            final_image_scores = vlm_hits
            final_beta = self.image_alpha
        
        final_hits = DenseRetriever.hybrid_top_k(
            bm25_scores=bm25_hits,
            dense_scores=dense_hits,
            image_scores=final_image_scores,
            caption_scores=caption_hits,
            alpha=self.alpha,
            beta=final_beta,
            gamma=self.caption_gamma if caption_hits is not None else 0.0,
            top_k=self.final_top_k,
            normalize=True,
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
            passage_ids = engine.search_single(query_text=query_text, doc_name=None, query_image_path=img_path, caption_prompt=None)

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
    vlm_retriever=None,
    passages_df=None,
    captioner: Optional[ImageCaptioner] = None,
    caption_gamma: float = 0.0,
    caption_top_k: int = 100,
    caption_append_to_query: bool = True
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
        captioner=captioner,
        caption_gamma=caption_gamma,
        caption_top_k=caption_top_k,
        caption_append_to_query=caption_append_to_query
    )

    output_path = Path(output_path)

    if mode == "m2kr":
        run_m2kr_hybrid(df_queries=queries, engine=engine, output_path=output_path)
    elif mode == "mmdoc":
        run_mmdoc_hybrid(queries=queries, engine=engine, output_path=output_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
