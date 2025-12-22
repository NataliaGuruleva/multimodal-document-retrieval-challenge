# Multimodal Document Retrieval

This repo contains a hybrid retrieval baseline for Kaggle **Multimodal Document Retrieval Challenge**:
- **M2KR:** query-image branch (CLIP), optional query-image captions (BLIP)
- **MMDoc/ViDoRe extras:** optional text to image retrieval (SigLIP VLM)

## Setup

```bash

pip install -r requirements.txt
```

Project structure:
- data/: datasets
- artifacts/: indices, embeddings, local eval outputs, submissions

## Expected data paths

### M2KR-Challenge (Kaggle)
```
data/M2KR-Challenge/challenge_passage/train-00000-of-00001.parquet
data/M2KR-Challenge/challenge_data/train-00000-of-00001.parquet
data/M2KR-Challenge/passage_images/
```

### M2KR-HF-local (M2KR local eval)
```
data/M2KR-HF-local/data/test-00000-of-00001.parquet
data/M2KR-HF-local/passages/test_passages-00000-of-00001.parquet
```

### MMDocIR-Challenge (Kaggle)
```
data/MMDocIR-Challenge/MMDocIR_pages.parquet
data/MMDocIR-Challenge/MMDocIR_gt_remove.jsonl
data/MMDocIR-Challenge/page_images/
```

### ViDoRe-DocVQA (local eval cache)
```
data/ViDoRe-DocVQA/pages.parquet
data/ViDoRe-DocVQA/queries.jsonl
data/ViDoRe-DocVQA/page_images/
```

## Local evaluations

### EVQA (M2KR-HF local)
```bash
python -m scripts.run_eval_evqa --alpha 0.4 --bm25_top_k 200 --dense_top_k 100 --final_top_k 5
```
Outputs:
- artifacts/local_eval/evqa_.../metrics.json
- artifacts/local_eval/evqa_.../predictions.jsonl

### ViDoRe (text-only or VLM text to image)
```bash
# text-only
python -m scripts.run_eval_vidore --use_vlm 0 --alpha 0.4

# with VLM (SigLIP text to image)
python -m scripts.run_eval_vidore --use_vlm 1 --alpha 0.4
```
Outputs:
- artifacts/local_eval/vidore_...jsonl

## Build Kaggle submissions

### M2KR submission (supports image-query and captions)
```bash
python -m scripts.run_m2kr --alpha 0.1 --beta 0.1 --gamma 0.0 --use_image 1 --use_caption 1
```

### MMDoc submission (supports VLM text to image)
```bash
python -m scripts.run_mmdoc --alpha 0.1 --use_vlm 1 --image_alpha 0.1
```

### Merge and convert to CSV
```bash
python -m scripts.merge_submissions
python -m scripts.conversion
```
Final file:
- artifacts/submission/submission.csv

