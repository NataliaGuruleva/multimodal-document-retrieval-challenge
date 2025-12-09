import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Sequence, Iterable
from rank_bm25 import BM25Okapi

# BM25 parameters
try:
    from config import BM25_K1, BM25_B
except ImportError:
    BM25_K1 = 1.5
    BM25_B = 0.75

Tokenizer = Callable[[str], List[str]]


class BM25DocumentIndex:
    """BM25 index for text search."""
    
    def __init__(self, bm25: BM25Okapi, doc_ids: List[str], doc_names: Optional[List[str]] = None):
        self.bm25 = bm25
        self.doc_ids = doc_ids
        self.doc_names = doc_names
    
    @classmethod
    def build(cls, doc_ids: Sequence[str], texts: Sequence[str], tokenizer: Tokenizer, 
              k1: float = BM25_K1, b: float = BM25_B, doc_names: Optional[Sequence[str]] = None) -> "BM25DocumentIndex":
        """Build BM25 index from document collection."""
        assert len(doc_ids) == len(texts), "doc_ids and texts must have same length"
        if doc_names is not None:
            assert len(doc_ids) == len(doc_names), "doc_ids and doc_names must have same length"
        
        # Tokenizing corpus
        tokenized_corpus = [tokenizer(text) for text in texts]
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        
        return cls(bm25=bm25, doc_ids=list(doc_ids), doc_names=list(doc_names) if doc_names is not None else None)
    
    def score(self, query: str, tokenizer: Tokenizer) -> List[float]:
        """Get BM25 scores for query across all documents."""
        query_tokens = tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)
        return list(scores)
    
    def retrieve(self, query: str, tokenizer: Tokenizer, top_k: int = 100, doc_name: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get top_k documents for query."""
        query_tokens = tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Filter by doc_name if provided
        if doc_name is not None and self.doc_names is not None:
            indices = [i for i, dn in enumerate(self.doc_names) if dn == doc_name]
        else:
            indices = range(len(scores))
        
        # Sort by score
        indices = sorted(indices, key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in indices]
    
    def save(self, path: str | Path) -> None:
        """Save index to pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("wb") as f:
            pickle.dump({"doc_ids": self.doc_ids, "bm25": self.bm25, "doc_names": self.doc_names}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, path: str | Path) -> "BM25DocumentIndex":
        """Load index from pickle file."""
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        
        return cls(bm25=data["bm25"], doc_ids=data["doc_ids"], doc_names=data.get("doc_names", None))
    
    def to_metadata_json(self) -> str:
        """Get metadata as JSON string."""
        meta = {"num_docs": len(self.doc_ids), "k1": getattr(self.bm25, "k1", None), "b": getattr(self.bm25, "b", None)}
        return json.dumps(meta, ensure_ascii=False, indent=2)


def batched_retrieve(index: BM25DocumentIndex, queries: Iterable[Tuple[str, str]], tokenizer: Tokenizer, top_k: int = 100) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """Batch retrieval for multiple queries."""
    results = []
    for qid, qtext in queries:
        hits = index.retrieve(qtext, tokenizer=tokenizer, top_k=top_k)
        results.append((qid, hits))
    
    return results
