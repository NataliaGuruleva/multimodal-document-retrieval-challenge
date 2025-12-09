from pathlib import Path
import json


ARTIFACTS_DIR = Path("artifacts/submission")
M2KR_FILE = ARTIFACTS_DIR / "submission_m2kr.jsonl"
MMDOC_FILE = ARTIFACTS_DIR / "submission_mmdoc.jsonl"
OUTPUT_FILE = ARTIFACTS_DIR / "submission_before_conversion_to_csv.jsonl"


def merge_jsonl(files, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for file in files:
            with file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    obj = json.loads(line.strip())
                    assert "question_id" in obj
                    assert "passage_id" in obj
                    assert isinstance(obj["passage_id"], list)
                    fout.write(json.dumps(obj) + "\n")
                    total += 1

    print(f"Merged {total} records into {output_path}")


def main():
    print("Merging submission files")

    merge_jsonl(
        files=[M2KR_FILE, MMDOC_FILE],
        output_path=OUTPUT_FILE,
    )

    print("Merge completed")


if __name__ == "__main__":
    main()
