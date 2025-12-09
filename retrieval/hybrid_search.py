from pathlib import Path
from typing import Dict, List, Iterable, Optional
import json
import pandas as pd
from tqdm import tqdm

from retrieval.bm25_index import BM25DocumentIndex
from retrieval.dense_retriever import DenseRetriever, DenseDocumentIndex
from retrieval.image_retriever import ImageIndex, ImageEncoder
from utils.text import build_query_text, tokenize


class HybridSearchEngine:
    """
    High-level hybrid search engine.
    - BM25 (sparse text)
    - Dense text retriever (sentence-transformers)
    """
    
    def __init__(
        self,
        bm25_index: BM25DocumentIndex,
        dense_index: DenseDocumentIndex,
        dense_retriever: DenseRetriever,
        alpha: float,
        bm25_top_k: int = 100,
        dense_top_k: int = 100,
        final_top_k: int = 5,
        # optional image branch
        image_index: Optional[ImageIndex] = None,
        image_encoder: Optional[ImageEncoder] = None,
        image_alpha: float = 0.3,
    ):
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.dense_retriever = dense_retriever
        
        # text fusion
        self.alpha = alpha
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k
        
        # image (queries)
        self.image_index = image_index
        self.image_encoder = image_encoder
        self.image_alpha = image_alpha
    
    def _has_image_branch(self) -> bool:

        return self.image_index is not None and self.image_encoder is not None
    
    def search_single(
        self,
        query_text: str,
        doc_name: Optional[str] = None,
        query_image_path: Optional[str] = None,
    ) -> List[str]:
        """
        Run hybrid retrieval for a single query.
        
        query_text : str
            Text query.
        doc_name : Optional[str]
            Document name (MMDocIR doc-restricted retrieval).
        query_image_path : Optional[str]
            Path to query image (img_path in M2KR).
            
        Returns Top passage_ids: List[str].
        """
        
        # BM25 sparse retrieval
        bm25_hits = self.bm25_index.retrieve(
            query=query_text,
            tokenizer=tokenize,
            top_k=self.bm25_top_k,
            doc_name=doc_name,
        )
        
        if not bm25_hits:
            return []
        
        candidate_ids = [pid for pid, _ in bm25_hits]
        
        # Dense text retrieval
        query_emb = self.dense_retriever.encode_text(query_text)
        dense_hits = self.dense_index.retrieve_subset(
            query_emb=query_emb,
            candidate_ids=candidate_ids,
            top_k=self.dense_top_k,
        )
        
        # Text hybrid (BM25 + Dense)
        text_hybrid_hits = DenseRetriever.hybrid_top_k(bm25_scores=bm25_hits, dense_scores=dense_hits, image_scores=None, alpha=self.alpha, beta=0.0, 
                                                       top_k=max(self.bm25_top_k, self.dense_top_k, self.final_top_k))
        
        # Image        
        if not self._has_image_branch():
            return [pid for pid, _ in text_hybrid_hits[:self.final_top_k]]        
        return [pid for pid, _ in text_hybrid_hits[:self.final_top_k]]


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
            img_path = None
            if "img_path" in row and isinstance(row["img_path"], str):
                img_path = row["img_path"]
            passage_ids = engine.search_single(query_text=query_text, doc_name=None, query_image_path=img_path)
            
            f.write(json.dumps({"question_id": qid, "passage_id": passage_ids}) + "\n")


def run_mmdoc_hybrid(queries: Iterable[Dict], engine: HybridSearchEngine, output_path: Path) -> None:
    """Run hybrid retrieval for MMDocIR"""
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
    )
    
    output_path = Path(output_path)
    
    if mode == "m2kr":
        run_m2kr_hybrid(df_queries=queries, engine=engine, output_path=output_path)
    elif mode == "mmdoc":
        run_mmdoc_hybrid(queries=queries, engine=engine, output_path=output_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")