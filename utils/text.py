import re
import html
import unicodedata
from typing import Iterable, List, Optional


_re_ctrl = re.compile(r"[\u0000-\u001F\u007F]")
_re_ws = re.compile(r"\s+")

def normalize_text(text: Optional[str]) -> str:
    """
    Basic text normalization (unicode-safe).
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _re_ctrl.sub(" ", text)
    text = _re_ws.sub(" ", text).strip()

    return text



_word_pattern = re.compile(r"[^\W_]+", flags=re.UNICODE)

def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Simple unicode tokenization (letters/digits, without underscore).
    """
    if not isinstance(text, str):
        return []
    if lowercase:
        text = text.lower()
    return _word_pattern.findall(text)


def concat_fields(values: Iterable[Optional[str]], sep: str = " ") -> str:
    cleaned = []
    for v in values:
        if isinstance(v, str) and v.strip():
            nv = normalize_text(v)
            if nv:
                cleaned.append(nv)
    return sep.join(cleaned)


def build_passage_text(ocr_text: Optional[str] = None, vlm_text: Optional[str] = None, fallback: Optional[str] = None) -> str:
    """
    Build passage text for retrieval.
    """
    if isinstance(vlm_text, str) and vlm_text.strip():
        return normalize_text(vlm_text)
    
    if isinstance(ocr_text, str) and ocr_text.strip():
        return normalize_text(ocr_text)
    
    return normalize_text(fallback)

_m2kr_labels = [
    "title",
    "hierarchical section title",
    "caption reference description",
    "caption attribution description",
    "content",
]

_m2kr_label_union = r"(?:title|hierarchical section title|caption reference description|caption attribution description|content)"

def _extract_m2kr_field(text: str, label: str) -> str:
    """
    Extracts a labeled block like 'title: ... content: ...'
    """
    pattern = rf"(?is)\b{re.escape(label)}\s*:\s*(.*?)(?=\b{_m2kr_label_union}\s*:|$)"
    m = re.search(pattern, text)
    if not m:
        return ""
    return normalize_text(m.group(1))

# boilerplate patterns that repeat across many passages
_m2kr_boilerplate = re.compile(
    r"(?is)"
    r"(this photo was created by.*?$|"
    r"it is not in the public domain.*?$|"
    r"use of this file outside of the licensing terms.*?$|"
    r"if you would like to use this image.*?$|"
    r"please credit authorship.*?$|"
    r"please maintain the original file name.*?$|"
    r"you can see a gallery.*?$|"
    r"if you have any questions.*?$|"
    r"wikimedia commons.*?$)"
)

def clean_m2kr_passage_content(passage_content: Optional[str], max_chars: int = 2500) -> str:
    """
    M2KR passage_content cleaner:
    - decodes html
    - removes repetitive licensing boilerplate
    - keeps only useful labeled fields: title/section/caption_ref/content
    - drops caption attribution description
    """
    t = normalize_text(passage_content)
    if not t:
        return ""

    # remove boilerplate
    t = _m2kr_boilerplate.sub(" ", t)
    t = normalize_text(t)

    title = _extract_m2kr_field(t, "title")
    section = _extract_m2kr_field(t, "hierarchical section title")
    caption_ref = _extract_m2kr_field(t, "caption reference description")
    content = _extract_m2kr_field(t, "content")

    # If labels are missing, fallback to whole cleaned text
    if not any([title, section, caption_ref, content]):
        out = t
    else:
        out = concat_fields([title, section, caption_ref, content])

    if max_chars is not None and max_chars > 0 and len(out) > max_chars:
        out = out[:max_chars]

    return out


def clean_m2kr_instruction(instruction: Optional[str]) -> str:
    """
    Drop M2KR instructions.
    """
    t = normalize_text(instruction)
    if not t:
        return ""
    low = t.lower()
    if "utilizing the given image" in low or ("given image" in low and "obtain documents" in low):
        return ""
    return t

def build_query_text(instruction: Optional[str] = None, question: Optional[str] = None) -> str:
    fields = []
    q = normalize_text(question)
    if q:
        fields.append(q)

    inst = clean_m2kr_instruction(instruction)
    if inst:
        fields.append(inst)

    return concat_fields(fields)

