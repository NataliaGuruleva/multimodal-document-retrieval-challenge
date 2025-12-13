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


def load_image(image_path: Optional[str] = None) -> Optional[Image.Image]:
    """
    Load image from path.
    """
    if not image_path or not isinstance(image_path, str):
        return None
    
    try:
        # Try absolute path
        p = Path(image_path)
        if p.exists():
            return Image.open(p).convert("RGB")
        
        # Try relative to DATA_DIR
        if DATA_DIR is not None:
            candidate = DATA_DIR / image_path
            if candidate.exists():
                return Image.open(candidate).convert("RGB")
        
        # Try as-is
        if p.exists():
            return Image.open(p).convert("RGB")     
        return None
    
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
    