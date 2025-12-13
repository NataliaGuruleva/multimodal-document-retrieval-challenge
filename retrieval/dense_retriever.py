import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Sequence, Iterable, Dict
import numpy as np
import torch


# Dense retriever parameters
try:
    from config import (
        DENSE_MODEL_NAME,
        EMB_BATCH_SIZE,
        DEVICE,
        DENSE_POOLING,
        HYBRID_ALPHA,
        HYBRID_BETA,
    )
except ImportError:
    DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMB_BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DENSE_POOLING = "mean"
    HYBRID_ALPHA = 0.5 # dense-text
    HYBRID_BETA = 0.3 # image


# sentence-transformers lazy load
_SentenceTransformer = None

def _lazy_load_model(model_name: str):
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer(model_name)


class DenseDocumentIndex:
    """Dense vector index for semantic search."""
    
    def __init__(self, doc_ids: List[str], embeddings: np.ndarray, model_name: str = DENSE_MODEL_NAME):
        self.doc_ids = doc_ids
        self.embeddings = embeddings
        self.model_name = model_name
    
    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])
    
    def _as_torch(self, device: str | torch.device = DEVICE) -> torch.Tensor:
        return torch.from_numpy(self.embeddings).to(device)
    
    def score(self, query_emb: np.ndarray, device: str | torch.device = DEVICE) -> np.ndarray:
        """Get similarity scores for query across all documents."""
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        
        q = torch.from_numpy(query_emb).to(device)
        d = self._as_torch(device)
        scores = torch.matmul(q, d.T)
        return scores.squeeze(0).cpu().numpy()
    
    def retrieve(self, query_emb: np.ndarray, top_k: int = 100, device: str | torch.device = DEVICE) -> List[Tuple[str, float]]:
        """Get top_k documents for query."""
        scores = self.score(query_emb, device=device)
        idx = np.argpartition(-scores, top_k)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.doc_ids[i], float(scores[i])) for i in idx.tolist()]
    
    def retrieve_subset(self, query_emb: np.ndarray, candidate_ids: Iterable[str], top_k: int) -> List[Tuple[str, float]]:
        """Retrieve only from a subset of document IDs."""
        hits = self.retrieve(query_emb, top_k=len(candidate_ids))
        cand = set(candidate_ids)
        hits = [(pid, s) for pid, s in hits if pid in cand]
        return hits[:top_k]
    
    def save(self, path: str | Path) -> None:
        """Save index to NPZ and metadata files."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(path.with_suffix(".npz"), embeddings=self.embeddings)
        meta = {
            "doc_ids": self.doc_ids,
            "model_name": self.model_name,
            "dim": int(self.embeddings.shape[1]),
            "num_docs": int(self.embeddings.shape[0]),
        }
        with path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "DenseDocumentIndex":
        """Load index from NPZ and metadata files."""
        path = Path(path)
        data = np.load(path.with_suffix(".npz"))
        with path.with_suffix(".meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(
            doc_ids=list(meta["doc_ids"]),
            embeddings=data["embeddings"],
            model_name=meta.get("model_name", DENSE_MODEL_NAME),
        )


class DenseRetriever:
    """Dense retriever for encoding texts and building indexes."""
    
    def __init__(self, model_name: str = DENSE_MODEL_NAME, device: str | torch.device = DEVICE, 
                 batch_size: int = EMB_BATCH_SIZE, show_progress: bool = True):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        self._model = _lazy_load_model(model_name)
        self._model.to(device)
    
    def encode_texts(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Encode multiple texts to embeddings."""
        all_embs: List[np.ndarray] = []
        it = range(0, len(texts), self.batch_size)
        
        if self.show_progress:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=f"Encoding texts ({self.model_name})")
        
        for start in it:
            batch = texts[start:start + self.batch_size]
            with torch.no_grad():
                embs = self._model.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
            all_embs.append(embs)
        return np.vstack(all_embs)
    
    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode single text to embedding."""
        return self.encode_texts([text], normalize)[0]
    
    def build_index(self, doc_ids: Sequence[str], texts: Sequence[str], normalize: bool = True) -> DenseDocumentIndex:
        """Build dense index from document collection."""
        assert len(doc_ids) == len(texts)
        embs = self.encode_texts(texts, normalize=normalize)
        return DenseDocumentIndex(list(doc_ids), embs, self.model_name)
    
    @staticmethod
    def hybrid_top_k(
        bm25_scores: List[Tuple[str, float]],
        dense_scores: List[Tuple[str, float]],
        image_scores: Optional[List[Tuple[str, float]]] = None,
        alpha: float = HYBRID_ALPHA,
        beta: float = HYBRID_BETA,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """
        Late fusion:
            score = (1 - alpha - beta) * bm25
                  + alpha * dense_text
                  + beta  * image
        
        Все скоры предполагаются нормированными или соизмеримыми.
        """
        assert 0.0 <= alpha <= 1.0
        assert 0.0 <= beta <= 1.0
        assert alpha + beta <= 1.0
        
        scores: Dict[str, float] = {}
        
        def add(src: Optional[List[Tuple[str, float]]], weight: float):
            if not src or weight == 0.0:
                return
            for pid, s in src:
                scores[pid] = scores.get(pid, 0.0) + weight * s
        
        add(bm25_scores, 1.0 - alpha - beta)
        add(dense_scores, alpha)
        add(image_scores, beta)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def batched_encode(retriever: DenseRetriever, texts: Iterable[Tuple[str, str]], normalize: bool = True) -> List[Tuple[str, np.ndarray]]:
    """Batch encoding for multiple texts."""
    results = []
    text_list = []
    id_list = []
    
    for tid, ttext in texts:
        text_list.append(ttext)
        id_list.append(tid)
    
    embeddings = retriever.encode_texts(text_list, normalize=normalize)
    
    for i, tid in enumerate(id_list):
        results.append((tid, embeddings[i]))
    
    return results