import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class ImageEncoder:
    """Vision encoder for image embeddings."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", batch_size: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        
        for p in self.model.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def encode_images(self, paths: List[Optional[str]]) -> torch.Tensor:
        """
        Encode images to normalized embeddings.
        """
        from utils.image import load_image
        
        n = len(paths)
        all_embs = []
        emb_dim = None
        
        for start in tqdm(range(0, n, self.batch_size), desc="Encoding images"):
            end = min(start + self.batch_size, n)
            batch_paths = paths[start:end]
            
            # Load valid images
            images = []
            valid_indices = []
            
            for i, p in enumerate(batch_paths):
                if not p:
                    continue
                img = load_image(image_path=p)
                if img is not None:
                    images.append(img)
                    valid_indices.append(i)
            
            # Handle empty batch
            if not images:
                if emb_dim is None:
                    all_embs.append(torch.zeros(len(batch_paths), 1))
                else:
                    all_embs.append(torch.zeros(len(batch_paths), emb_dim))
                continue
            
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            # Get embeddings
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embs = outputs.pooler_output
            else:
                embs = outputs.last_hidden_state.mean(dim=1)
            
            embs = F.normalize(embs, dim=-1)
            if emb_dim is None:
                emb_dim = int(embs.size(-1))

                if all_embs and all_embs[0].size(-1) == 1:
                    all_embs = [torch.zeros(t.size(0), emb_dim) for t in all_embs]
            batch_embs = torch.zeros(len(batch_paths), emb_dim)
            for idx, emb in zip(valid_indices, embs.cpu()):
                batch_embs[idx] = emb
            
            all_embs.append(batch_embs)
        embeddings = torch.cat(all_embs, dim=0)
        assert embeddings.size(0) == n
        
        return embeddings


class ImageIndex:
    """Index for image embeddings."""
    def __init__(self, id2idx: Dict[str, int], embeddings: torch.Tensor):
        self.id2idx = id2idx
        self.embeddings = embeddings
        self.dim = embeddings.shape[1]
    
    @classmethod
    def build_from_paths(cls, encoder: ImageEncoder, id_to_path: Dict[str, Optional[str]]) -> "ImageIndex":
        """Build index from id -> image path mapping."""
        ids_sorted = sorted(id_to_path.keys())
        paths = [id_to_path[i] for i in ids_sorted]
        
        embeddings = encoder.encode_images(paths)
        id2idx = {pid: i for i, pid in enumerate(ids_sorted)}
        
        return cls(id2idx=id2idx, embeddings=embeddings)
    
    def save(self, path: str | Path) -> None:
        """Save index to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({"id2idx": self.id2idx, "embeddings": self.embeddings}, path)
    
    @classmethod
    def load(cls, path: str | Path) -> "ImageIndex":
        """Load index from file."""
        path = Path(path)
        data = torch.load(path, map_location="cpu")
        return cls(id2idx=data["id2idx"], embeddings=data["embeddings"])
    
    def get_embeddings(self, ids: List[str]) -> torch.Tensor:
        """Get embeddings for specific IDs."""
        result = torch.zeros(len(ids), self.dim)
        
        for i, pid in enumerate(ids):
            idx = self.id2idx.get(pid)
            if idx is not None:
                result[i] = self.embeddings[idx]
        
        return result


def image_retrieval_scores(query_embeddings: torch.Tensor, passage_embeddings: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    """
    Compute cosine similarity between queries and passages.
    
    Args:
        query_embeddings: [Q, D]
        passage_embeddings: [P, D]
    
    Returns:
        similarity scores: [Q, P]
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    q = query_embeddings.to(device)
    p = passage_embeddings.to(device)
    scores = torch.matmul(q, p.t())
    return scores.cpu()
