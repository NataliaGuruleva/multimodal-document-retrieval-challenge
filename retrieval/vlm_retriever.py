from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from tqdm.auto import tqdm

from utils.text import normalize_text
from utils.image import load_image



class VLMRetriever:
    """
    VLM reranker based on SigLIP.
    """

    def __init__(
        self,
        model_name: str = "C:/Users/dimaa/siglip_model",
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size

        print(f"Loading {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)
        self._img_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        text = normalize_text(text)
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        out = self.model.get_text_features(**inputs)
        out = F.normalize(out, dim=-1)
        
        return out

    @torch.no_grad()
    def encode_candidates(self, texts: List[str], image_paths: List[Optional[str]]) -> torch.Tensor:
        """
        Encodes ONLY images.
        """
        return self.encode_images(image_paths)

    @torch.no_grad()
    def encode_images(self, image_paths: List[Optional[str]]) -> torch.Tensor:
        """
        Encode images into normalized embeddings.
        """
        n = len(image_paths)
        if n == 0:
            return torch.empty(0, self.model.config.vision_config.hidden_size, device=self.device)
        out = torch.zeros(n, self.model.config.vision_config.hidden_size, device=self.device)

        B = self.batch_size
        for start in tqdm(range(0, n, B), desc="Image encode"):
            end = min(start + B, n)
            batch_paths = image_paths[start:end]
            images = []
            valid_pos = []
            cached_pos = []

            for j, p in enumerate(batch_paths):
                idx = start + j
                if not p or not isinstance(p, str):
                    continue
                if p in self._img_cache:
                    out[idx] = self._img_cache[p].to(self.device)
                    cached_pos.append(idx)
                    continue
                img = load_image(p)
                if img is None:
                    continue
                images.append(img)
                valid_pos.append(idx)

            if images:
                inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
                img_feat = self.model.get_image_features(**inputs)
                img_feat = F.normalize(img_feat, dim=-1)
                for k, idx in enumerate(valid_pos):
                    v = img_feat[k]
                    out[idx] = v
                    p = image_paths[idx]
                    if isinstance(p, str):
                        self._img_cache[p] = v.detach().cpu()

        return out


    @torch.no_grad()
    def retrieve(self, query_text: str, doc_page_ids: List[str], image_paths: List[Optional[str]], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Text -> Page screenshots retrieval for MMDocIR.
        """
        assert len(doc_page_ids) == len(image_paths)
        q = self.encode_query(query_text)
        pages = self.encode_images(image_paths)
        scores = (q @ pages.T).squeeze(0)
        k = min(int(top_k), int(scores.numel()))
        if k <= 0:
            return []

        idx = torch.topk(scores, k).indices.tolist()
        return [(doc_page_ids[i], float(scores[i])) for i in idx]
