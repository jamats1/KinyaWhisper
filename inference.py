#!/usr/bin/env python3
"""
test_model.py

Performs transcription on audio samples using a fine-tuned Whisper model.
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os

model_dir = "kinya-whisper-model"
audio_dir = "audio"
output_file = "transcriptions.txt"

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
processor = WhisperProcessor.from_pretrained(model_dir)

model.eval()
device = torch.device("cpu")

with open(output_file, "w") as f_out:
    for i in range(1, 103):
        filename = f"{i:03}.wav"
        filepath = os.path.join(audio_dir, filename)
        print(f"🔍 Transcribing {filename}...")

        waveform, sample_rate = torchaudio.load(filepath)
        inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                max_length=64,
                num_beams=5,
                do_sample=False,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                early_stopping=True
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"{filename}: {transcription}")
        f_out.write(f"{filename}: {transcription}\n")