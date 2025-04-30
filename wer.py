from jiwer import wer
import json

# === Step 1: Load expected transcriptions ===
with open("dataset.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

expected = [entry["text"].strip() for entry in dataset]

# === Step 2: Load actual transcriptions ===
actual = []
with open("transcriptions.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(": ", 1)
        if len(parts) == 2:
            actual.append(parts[1].strip())
        else:
            actual.append("")

# === Step 3: Validate and compute ===
if len(expected) != len(actual):
    print(f"❌ Mismatch: {len(expected)} expected vs {len(actual)} actual")
else:
    error_rate = wer(expected, actual)
    print(f"✅ WER: {error_rate:.2%}")