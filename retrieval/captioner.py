import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from tqdm.auto import tqdm
from utils.image import load_image
from utils.text import normalize_text

warnings.filterwarnings("ignore")


class ImageCaptioner:
    """
    Image captioner for M2KR.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        batch_size: int = 8,
        device: Optional[str] = None,
        max_new_tokens: int = 32,
        num_beams: int = 3,
        show_progress: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.show_progress = show_progress

        from transformers import BlipProcessor, BlipForConditionalGeneration

        print(f"Loading captioner: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        # cache: path -> caption
        self._cache: dict[str, str] = {}

    def _clean_prompt(self, prompt: Optional[str]) -> Optional[str]:
        if not isinstance(prompt, str):
            return None
        p = normalize_text(prompt)
        if not p or p == "Empty.":
            return None
        # Drop meta-instructions
        low = p.lower()
        if "provide a brief description" in low or "describe the image" in low:
            return None
        if len(p) > 128:
            p = p[:128]
        return p

    @torch.no_grad()
    def caption_images(self, paths: Sequence[Optional[str]], prompts: Optional[Sequence[Optional[str]]] = None) -> List[str]:
        n = len(paths)
        out: List[str] = [""] * n

        if prompts is None:
            prompts = [None] * n
        assert len(prompts) == n

        for start in tqdm(range(0, n, self.batch_size), desc="Captioning images"):
            end = min(start + self.batch_size, n)
            batch_paths = paths[start:end]
            batch_prompts = prompts[start:end]
            images = []
            valid_idx = []
            valid_prompts = []
            for j, (p, pr) in enumerate(zip(batch_paths, batch_prompts)):
                if not p or not isinstance(p, str):
                    continue
                if p in self._cache:
                    out[start + j] = self._cache[p]
                    continue
                img = load_image(p)
                if img is None:
                    continue
                images.append(img)
                valid_idx.append(start + j)
                valid_prompts.append(self._clean_prompt(pr))

            if not images:
                continue

            # If all prompts are None, do pure captioning
            has_prompt = any(isinstance(x, str) and x.strip() for x in valid_prompts)

            if has_prompt:
                inputs = self.processor(images=images, text=valid_prompts, return_tensors="pt", padding=True).to(self.device)
            else:
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams, do_sample=False)
            caps = self.processor.batch_decode(output_ids, skip_special_tokens=True)

            for idx, cap in zip(valid_idx, caps):
                cap = normalize_text(cap)
                if cap == "Empty.":
                    cap = ""
                out[idx] = cap
                # cache
                pth = paths[idx]
                if isinstance(pth, str):
                    self._cache[pth] = cap

        return out

    def caption_image(self, path: Optional[str], prompt: Optional[str] = None) -> str:
        if not path or not isinstance(path, str):
            return ""
        return self.caption_images([path], prompts=[prompt])[0]
