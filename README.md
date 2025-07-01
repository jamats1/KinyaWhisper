## 🗣️ KinyaWhisper
KinyaWhisper is a fine-tuned version of OpenAI's Whisper model for automatic speech recognition (ASR) in Kinyarwanda. It was trained on 102 manually labeled .wav files and serves as a reproducible baseline for speech recognition in low-resource, indigenous languages.

## 🤗 Hugging Face Model

The fine-tuned KinyaWhisper model is publicly available on Hugging Face:

➡️ [https://huggingface.co/benax-rw/KinyaWhisper](https://huggingface.co/benax-rw/KinyaWhisper)

You can use it directly in your code:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load fine-tuned KinyaWhisper model and processor from Hugging Face
model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("your_audio.wav")
inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

# Generate prediction
predicted_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("🗣️ Transcription:", transcription)
```

## 🏋️ Taining Details
•	Model: openai/whisper-small
•	Epochs: 80
•	Batch size: 4
•	Learning rate: 1e-5
•	Optimizer: Adam
•	Final loss: 0.00024
•	WER: 51.85%

## ⚠️Limitations
The model was trained on a small dataset (102 samples). It performs best on short, clear Kinyarwanda utterances and may struggle with longer or noisy audio. This is an early-stage educational model, not yet suitable for production use.

## 📚 Citation

If you use this model, please cite:

```bibtex
@misc{baziramwabo2025kinyawhisper,
  author       = {Gabriel Baziramwabo},
  title        = {KinyaWhisper: Fine-Tuning Whisper for Kinyarwanda ASR},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/benax-rw/KinyaWhisper}},
  note         = {Version 1.0}
}
```
## 📬 Contact
Maintained by Gabriel Baziramwabo. 
✉️ gabriel@benax.rw
�� https://benax.rw

## 🚀 Deployment (Coolify / Docker)

This repository includes a production-ready Dockerfile tested on Coolify.
Only the libraries actually required at runtime are installed to keep image size and build time low.

1. `requirements-prod.txt` lists the **5** necessary packages:
   - Flask
   - Werkzeug
   - torch
   - torchaudio
   - transformers

2. The Dockerfile now copies and installs this file directly:
   ```dockerfile
   COPY requirements-prod.txt ./
   RUN pip install --upgrade pip && \
       pip install --no-cache-dir -r requirements-prod.txt
   ```

To build locally:
```bash
docker build -t kinya-whisper .
```

Then run:
```bash
docker run -p 5000:5000 kinya-whisper
```

These changes were introduced via Shrimp tasks to resolve Coolify deployment failures caused by platform-specific dependencies (e.g., `tensorflow-macos`).
