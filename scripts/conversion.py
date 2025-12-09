from pathlib import Path
import json
import csv


SUBMISSION_DIR = Path("artifacts/submission")

INPUT_JSONL = SUBMISSION_DIR / "submission_before_conversion_to_csv.jsonl"
OUTPUT_CSV = SUBMISSION_DIR / "submission.csv"


def jsonl_to_csv(jsonl_path: Path, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("r", encoding="utf-8") as fin, \
         csv_path.open("w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout)
        writer.writerow(["question_id", "passage_id"])

        total = 0
        for line in fin:
            obj = json.loads(line.strip())

            qid = str(obj["question_id"])
            passage_ids = json.dumps([str(pid) for pid in obj["passage_id"]])

            writer.writerow([qid, passage_ids])
            total += 1

    print(f"Converted {total} rows to {csv_path}")


def main():
    print("Converting JSONL to CSV...")
    jsonl_to_csv(INPUT_JSONL, OUTPUT_CSV)
    print("submission.csv saved")


if __name__ == "__main__":
    main()
