import torch
import io
from pathlib import Path
from typing import Optional
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_image_path(image_path: Optional[str]) -> Optional[Path]:
    """
    Robustly resolve image paths across:
      - absolute paths
      - paths relative to cwd / project root
      - paths relative to DATA_DIR
      - dataset subfolders (M2KR/MMDocIR/ViDoRe)
      - duplicated prefixes (we fallback to basename search)
    """
    if not image_path or not isinstance(image_path, str):
        return None

    p = Path(image_path)
    name = p.name

    candidates: list[Path] = []

    # as absolute or relative to current
    candidates.append(p)
    # relative to project root
    candidates.append(PROJECT_ROOT / p)

    if DATA_DIR is not None:
        # relative to data/
        candidates.append(DATA_DIR / p)

        # common dataset roots
        for root in ["M2KR-Challenge", "MMDocIR-Challenge/page_images", "ViDoRe-DocVQA", "M2KR-HF-local"]:
            candidates.append(DATA_DIR / root / p)

        # common image folders
        for folder in [
            "M2KR-Challenge/passage_images",
            "MMDocIR-Challenge/page_images",
            "ViDoRe-DocVQA/page_images",
        ]:
            candidates.append(DATA_DIR / folder / name)

    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            continue
    return None

def load_image(image_path: Optional[str] = None) -> Optional[Image.Image]:
    """
    Load image from path.
    """
    if not image_path or not isinstance(image_path, str):
        return None
    
    try:
        rp = resolve_image_path(image_path)
        if rp is None:
            return None
        return Image.open(rp).convert("RGB")
    
    except Exception:
        return None


def load_image_from_bytes(image_binary: Optional[bytes]) -> Optional[Image.Image]:
    """
    Load image from binary data.
    """
    if not image_binary:
        return None
    try:
        return Image.open(io.BytesIO(image_binary)).convert("RGB")
    except Exception:
        return None


def get_dummy_vision_tensor(device: str = "cpu", image_size: int = 224) -> torch.Tensor:

    return torch.zeros((1, 1, 1, 3, image_size, image_size), device=device, dtype=torch.float32)


def prepare_vision_input(image: Optional[Image.Image], image_processor, device: str = "cpu", image_size: int = 224) -> torch.Tensor:
    """
    Prepare image tensor for vision models.
    
    Returns:
        Tensor shape (1, 1, 1, 3, H, W)
    """
    
    if image is None:
        return get_dummy_vision_tensor(device, image_size)  
    try:
        img_tensor = image_processor(image)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device)
        return img_tensor
    except Exception:
        return get_dummy_vision_tensor(device, image_size)
    