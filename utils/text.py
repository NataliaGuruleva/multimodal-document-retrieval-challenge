import re
from typing import Iterable, List, Optional


def normalize_text(text: Optional[str]) -> str:
    """
    Basic text normalization.
    """
    if not isinstance(text, str):
        return "Empty."
    
    # remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text if text else "Empty."


_word_pattern = re.compile(r"\b\w+\b")

def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Simple tokenization.
    """
    if lowercase:
        text = text.lower()
    
    return _word_pattern.findall(text)


def concat_fields(values: Iterable[Optional[str]], sep: str = " ") -> str:
    """
    Concatenates multiple texts into a single.
    - instruction + question (M2KR queries)
    - ocr_text + vlm_text (MMDocIR passages)
    
    Empty values are skipped.
    """
    cleaned = [normalize_text(v) for v in values if isinstance(v, str) and v.strip()]
    if not cleaned:
        return "Empty."
    
    return sep.join(cleaned)


def build_query_text(instruction: Optional[str] = None, question: Optional[str] = None) -> str:
    """
    Build query text for retrieval.
    M2KR:
        instruction + question
    MMDocIR:
        question only
    """
    fields = []
    if instruction:
        fields.append(instruction)
    if question:
        fields.append(question)
    
    return concat_fields(fields)


def build_passage_text(ocr_text: Optional[str] = None, vlm_text: Optional[str] = None, fallback: Optional[str] = None) -> str:
    """
    Build passage text for retrieval.
    """
    if isinstance(vlm_text, str) and vlm_text.strip():
        return normalize_text(vlm_text)
    
    if isinstance(ocr_text, str) and ocr_text.strip():
        return normalize_text(ocr_text)
    
    return normalize_text(fallback)