from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os
from pathlib import Path

# ====== Configuration ======
audio_path = "unseen_audio_data/rw-unseen-001.mp3"
output_dir = "transcription_output"

# ====== Load model and processor ======
model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")

# ====== Load and preprocess audio ======
waveform, sample_rate = torchaudio.load(audio_path)

# Convert stereo to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)

# Prepare input
inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(inputs["input_features"])

# Decode transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcription:", transcription)

# ====== Prepare output file path ======
input_filename = Path(audio_path).stem  # e.g., "rw-unseen001"
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)  # Create folder if not exist
output_file = output_path / f"{input_filename}.txt"

# Save to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(transcription)

print(f"Transcription saved to '{output_file}'")