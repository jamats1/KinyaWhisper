---
license: mit
language:
- rw
pretty_name: KinyaWhisper
task_categories:
- automatic-speech-recognition
tags:
- rw
- Kin
- Kinyarwanda
- Rwanda
- ururimi
- rwacu
- ururimi rwacu
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: text
    dtype: string
  - name: audio_len
    dtype: float64
  - name: transcript_len
    dtype: int64
  - name: len_ratio
    dtype: float64
  splits:
  - name: train
    num_bytes: 2475947.0
    num_examples: 102
  download_size: 2478359
  dataset_size: 2475947.0
---
# Kinyarwanda Spoken Words Dataset

This dataset contains 102 short audio samples of spoken Kinyarwanda words, each labeled with its corresponding transcription. It is designed for training, evaluating, and experimenting with Automatic Speech Recognition (ASR) models in low-resource settings.

## Structure

- `audio/`: Contains 102 `.wav` files (mono, 16kHz)
- `transcripts.txt`: Tab-separated transcription file (e.g., `001.wav\tmuraho`)
- `manifest.jsonl`: JSONL file with audio paths and text labels (compatible with 🤗 Datasets and Whisper training scripts)

## Example

```json
{"audio_filepath": "audio/001.wav", "text": "muraho"}
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("benax-rw/my_kinyarwanda_dataset", split="train")
example = ds[0]
print(example["audio"]["array"], example["text"])
```

## License

This dataset is published for educational and research purposes.

## Citation

If you use this dataset, please cite:
> Benax Labs, KinyaWhisper Dataset for Fine-tuning Whisper on Kinyarwanda (2025)