from pathlib import Path
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from PIL import Image
from tqdm.auto import tqdm
from utils.text import normalize_text


class VLMRetriever:
    """
    VLM reranker based on SigLIP.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
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

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        text = normalize_text(text)

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self.device)

        out = self.model.get_text_features(**inputs)
        out = F.normalize(out, dim=-1)
        return out

    @torch.no_grad()
    def encode_candidates(
        self,
        texts: List[str],
        image_paths: List[Optional[str]],
    ) -> torch.Tensor:

        embs = []
        B = self.batch_size

        for start in tqdm(range(0, len(texts), B), desc="SigLIP rerank"):
            end = min(start + B, len(texts))

            batch_texts = [normalize_text(t) for t in texts[start:end]]
            batch_images = [
                Image.open(p).convert("RGB") if p and Path(p).exists() else None
                for p in image_paths[start:end]
            ]

            inputs = self.processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            ).to(self.device)

            img_feat = self.model.get_image_features(**inputs)
            img_feat = F.normalize(img_feat, dim=-1)
            embs.append(img_feat)
        if len(embs) == 0:
            return torch.empty(0, self.model.config.text_config.hidden_size, device=self.device)

        return torch.cat(embs, dim=0)

    @torch.no_grad()
    def retrieve(
        self,
        query_text: str,
        candidate_texts: List[str],
        candidate_image_paths: List[Optional[str]],
        candidate_ids: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:

        q = self.encode_query(query_text)
        c = self.encode_candidates(candidate_texts, candidate_image_paths)

        scores = (q @ c.T).squeeze(0).cpu()
        k = min(top_k, scores.numel())
        idx = torch.topk(scores, k).indices.tolist()

        return [(candidate_ids[i], float(scores[i])) for i in idx]
