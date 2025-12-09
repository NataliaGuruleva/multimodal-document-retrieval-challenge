import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class VLMRetriever:
    """Vision-Language retriever for text queries and image+text candidates."""
    def __init__(self, model_name: str = "BAAI/BGE-VL-MLLM-S2", batch_size: int = 1, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.set_processor(model_name)
    
    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        """Encode text query."""
        from utils.text import normalize_text
        text = normalize_text(text)
        inputs = self.model.data_process(text=text, images=None, q_or_c="q")        
        emb = self.model(**inputs, output_hidden_states=True)[:, -1, :]
        emb = F.normalize(emb, dim=-1)
        return emb
    
    @torch.no_grad()
    def encode_candidates(self, texts: List[str], image_paths: List[Optional[str]]) -> torch.Tensor:
        """Encode candidates with text and optional images."""
        from utils.image import load_image
        from utils.text import normalize_text
        assert len(texts) == len(image_paths)
        embeddings = []
        
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Encoding VLM"):
            end = min(start + self.batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_paths = image_paths[start:end]
            
            # Load images
            images = []
            for p in batch_paths:
                img = load_image(image_path=p) if p else None
                images.append(img)

            batch_texts = [normalize_text(t) for t in batch_texts]
            inputs = self.model.data_process(text=batch_texts, images=images, q_or_c="c")
            
            # Get embeddings
            batch_emb = self.model(**inputs, output_hidden_states=True)[:, -1, :]
            batch_emb = F.normalize(batch_emb, dim=-1)
            embeddings.append(batch_emb)
        return torch.cat(embeddings, dim=0)
    
    def retrieve(self, query_text: str, candidate_texts: List[str], candidate_image_paths: List[Optional[str]], 
                 candidate_ids: List[str], top_k: int = 5,) -> List[Tuple[str, float]]:
        """Retrieve top_k candidates for query."""
        assert len(candidate_texts) == len(candidate_ids)
        assert len(candidate_texts) == len(candidate_image_paths)
        
        # query
        query_emb = self.encode_query(query_text).to(self.device)
        
        # candidates
        candidate_embs = self.encode_candidates(candidate_texts, candidate_image_paths).to(self.device)
        
        # similarity scores
        scores = torch.matmul(query_emb, candidate_embs.T).squeeze(0).cpu()
        
        # Get top_k indices
        top_k = min(top_k, scores.numel())
        top_indices = torch.topk(scores, top_k).indices.tolist()
        
        # Return results
        return [(candidate_ids[i], float(scores[i])) for i in top_indices]
    